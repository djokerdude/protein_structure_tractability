"""Anthropic tool-use agent for protein structure tractability assessment.

Four phases, strictly separated:

1. Entity resolution  — LLM reasons over search candidates, optionally prompts
                        the user, and emits one resolved UniProt accession.
                        This is the only phase that genuinely needs LLM judgment
                        (heterogeneous candidates, organism/isoform ambiguity).

2. Data acquisition   — Pure Python: call each tool function sequentially with
                        the resolved accession.  No LLM in the loop.

3. Deterministic math — compute.py, scoring.py, qc.py produce every number that
                        appears in the report.  LLM is not involved.

3b. Purification      — Fetch PubMed abstracts for high-res structures, then one
                        structured LLM call extracts protocol fields.  The LLM
                        writes only `notes` (one grounded sentence); all other
                        fields are validated against controlled vocabularies and
                        scored by pure code in purification.py.

4. Narrative          — One final LLM call, handed the complete fact sheet.
                        Writes *only* ``reasoning`` and ``recommended_strategy``.
                        The prompt forbids producing any number not already in
                        the fact sheet.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

import anthropic
from pydantic import BaseModel

from . import compute, qc, scoring
from .purification import purification_tractability_score
from .schema import (
    Domain,
    ExperimentalStructure,
    ExpressionSystem,
    Provenance,
    PurificationProtocol,
    QCFlag,
    ResidueRange,
    ScoreBreakdown,
    Source,
    TractabilityReport,
    YieldCategory,
)
from .tools import (
    UniProtEntry,
    get_alphafold_plddt,
    get_ncbi_record,
    get_pdb_structures,
    get_pubmed_abstracts,
    get_sifts_coverage,
    get_uniprot_entry,
    search_uniprot,
)

_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 4096

# ── Resolution-phase tool schemas ─────────────────────────────────────────────

_RESOLUTION_TOOLS: list[dict] = [
    {
        "name": "search_uniprot",
        "description": (
            "Search UniProt for protein candidates matching a free-text protein name or "
            "gene symbol.  Returns ranked candidates with accession, protein name, "
            "organism, sequence length, and whether the entry is reviewed (Swiss-Prot).  "
            "Always call this before selecting a target unless the user supplied a bare "
            "UniProt accession (e.g. 'P38398')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text protein name, gene symbol, or accession.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum candidates to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Present candidates to the user and request a selection when the search "
            "returns multiple plausible, non-equivalent targets (e.g. same gene in "
            "different organisms, canonical vs. isoform, two unrelated proteins with "
            "the same common name).  Do NOT call this if the correct target is clear "
            "from context — prefer Homo sapiens reviewed entries for unqualified "
            "biomedical queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A clear, specific question for the user.",
                },
                "candidates": {
                    "type": "array",
                    "description": "Candidates to display.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "accession": {"type": "string"},
                            "protein_name": {"type": "string"},
                            "organism": {"type": "string"},
                            "sequence_length": {"type": "integer"},
                            "reviewed": {"type": "boolean"},
                        },
                        "required": ["accession", "protein_name", "organism"],
                    },
                },
            },
            "required": ["question", "candidates"],
        },
    },
    {
        "name": "select_accession",
        "description": (
            "Signal that entity resolution is complete.  Call this once you have "
            "identified the best UniProt accession.  You must have called "
            "search_uniprot at least once before calling this (or the user supplied "
            "the accession directly)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "accession": {
                    "type": "string",
                    "description": "Resolved UniProt accession (e.g. 'P38398').",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "Confidence: 'high' = reviewed entry, unambiguous match; "
                        "'medium' = minor uncertainty; 'low' = best guess."
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": "Optional one-line rationale for the selection.",
                },
            },
            "required": ["accession", "confidence"],
        },
    },
]

_RESOLUTION_SYSTEM = """\
You are the entity-resolution step of a protein structure tractability pipeline.

Goal: identify the single best UniProt accession for the protein the user named.

Protocol:
- If the user supplied a raw UniProt accession (format: [A-Z][0-9][A-Z0-9]{3}[0-9]
  or similar), call select_accession directly without searching.
- Otherwise, call search_uniprot first.
- Prefer reviewed (Swiss-Prot) entries over unreviewed (TrEMBL).
- For biomedical targets with no organism specified, prefer Homo sapiens.
- If candidates are genuinely ambiguous (different proteins, not just different
  organisms for the same well-known gene), call ask_user.
- Once confident, call select_accession.  Do not end the turn without calling it.
"""


# ── Purification extraction models ────────────────────────────────────────────

class _SingleProtocolExtraction(BaseModel):
    """LLM-filled fields for one PDB structure's purification protocol."""

    pdb_id: str
    expression_system: ExpressionSystem
    purification_steps: list[str]
    requires_coexpression: bool = False
    yield_category: YieldCategory = "unknown"
    construct_description: str
    notes: str  # one-line grounded summary — only LLM-authored field


class _PurificationExtractionBatch(BaseModel):
    protocols: list[_SingleProtocolExtraction]


_PURIFICATION_EXTRACTION_SYSTEM = """\
You are extracting structured purification protocol information from PubMed abstracts
of protein crystallography and cryo-EM papers.

For each abstract provided, extract:

  pdb_id                 — the PDB ID given in the header (copy it exactly)
  expression_system      — host organism used: "ecoli", "insect", "mammalian",
                           "yeast", "cell_free", or "unknown"
  purification_steps     — ordered list of chromatography/purification steps mentioned
                           (e.g. ["Ni-NTA affinity", "anion exchange", "size exclusion"])
  requires_coexpression  — true if the paper states the protein needed a partner
                           protein or chaperone for solubility or stability
  yield_category         — "high" / "medium" / "low" / "unknown" based on any
                           explicit quantity or yield statement in the abstract
  construct_description  — the protein construct described (domain boundaries, tags,
                           truncations, mutations), e.g. "TIR domain (560–724), His6-tag"
  notes                  — one concise sentence summarising the key purification
                           challenge or notable feature, grounded strictly in the abstract

Rules:
- Extract only information explicitly stated in the abstract.  Do not infer.
- If a field is not mentioned, use "unknown" / false / empty list as appropriate.
- Include exactly one protocol entry per PDB ID provided in the input, even if
  the abstract is short or lacks detail — fill missing fields with unknowns.
- Keep notes to one sentence and base it solely on the abstract text.
"""


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialize_candidates(candidates: list) -> str:
    return json.dumps(
        [
            {
                "accession": c.accession,
                "protein_name": c.protein_name,
                "gene": c.gene,
                "organism": c.organism,
                "sequence_length": c.sequence_length,
                "reviewed": c.reviewed,
            }
            for c in candidates
        ]
    )


# ── Phase 1: Entity resolution ────────────────────────────────────────────────

def _run_resolution_tool(
    name: str, inputs: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Execute one resolution-phase tool call.

    Returns ``(result_json, selection)`` where ``selection`` is non-None only
    for ``select_accession``, signalling the loop to stop.
    """
    if name == "search_uniprot":
        candidates = search_uniprot(inputs["query"], inputs.get("limit", 10))
        return _serialize_candidates(candidates), None

    if name == "ask_user":
        print(f"\n{inputs['question']}\n")
        for i, c in enumerate(inputs.get("candidates", []), 1):
            reviewed_tag = " [reviewed]" if c.get("reviewed") else ""
            print(
                f"  {i}.  {c.get('accession', '?'):10s}  "
                f"{c.get('protein_name', '')[:50]:<50s}  "
                f"{c.get('organism', '')}{reviewed_tag}"
            )
        print()
        user_answer = input("Enter your choice (number or accession): ").strip()
        return json.dumps({"user_response": user_answer}), None

    if name == "select_accession":
        return json.dumps({"accepted": True}), inputs

    return json.dumps({"error": f"Unknown resolution tool: {name}"}), None


def resolve_protein(query: str, client: anthropic.Anthropic) -> tuple[str, str]:
    """Run the entity-resolution agent loop.

    Returns ``(accession, notes)`` where *notes* is the agent's one-line
    selection rationale.  Raises ``ValueError`` if resolution fails.
    """
    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Resolve this protein to a UniProt accession: {query!r}",
        }
    ]

    while True:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_RESOLUTION_SYSTEM,
            tools=cast(Any, _RESOLUTION_TOOLS),
            messages=cast(Any, messages),
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            raise ValueError(
                f"Entity resolution ended without a select_accession call for {query!r}. "
                "The system prompt requires the agent to always call select_accession."
            )

        if response.stop_reason != "tool_use":
            raise ValueError(
                f"Unexpected stop_reason during entity resolution: {response.stop_reason!r}"
            )

        tool_results: list[dict] = []
        selection: dict[str, Any] | None = None

        for block in response.content:
            if block.type != "tool_use":
                continue
            result_str, sel = _run_resolution_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                }
            )
            if sel is not None:
                selection = sel

        messages.append({"role": "user", "content": tool_results})

        if selection is not None:
            accession = selection["accession"].strip().upper()
            notes = selection.get("notes", "")
            return accession, notes


# ── Phase 2: Data acquisition ─────────────────────────────────────────────────

def acquire_data(accession: str) -> dict[str, Any]:
    """Fetch all data sources for a resolved accession.

    AlphaFold pLDDT and NCBI record are best-effort: failures are silently
    captured as ``None`` so the rest of the pipeline can proceed.
    """
    entry = get_uniprot_entry(accession)
    structures = get_pdb_structures(accession)
    covered_ranges = get_sifts_coverage(accession)

    try:
        plddt: list[float] | None = get_alphafold_plddt(accession)
    except Exception:
        plddt = None

    try:
        ncbi: dict | None = get_ncbi_record(accession)
    except Exception:
        ncbi = None

    return {
        "entry": entry,
        "structures": structures,
        "covered_ranges": covered_ranges,
        "plddt": plddt,
        "ncbi": ncbi,
    }


# ── Phase 3: Deterministic compute ────────────────────────────────────────────

def compute_facts(
    entry: UniProtEntry,
    structures: list[ExperimentalStructure],
    covered_ranges: list[tuple[int, int]],
    plddt: list[float] | None,
) -> dict[str, Any]:
    """Run all deterministic geometry and produce a compact fact dict.

    Nothing here calls the network or the LLM.  Every number in the final
    report originates from this function or from scoring.score().
    """
    seq_len = entry.sequence_length
    disordered: list[tuple[int, int]] = list(entry.disordered_regions)

    cov_frac = compute.coverage_fraction(covered_ranges, seq_len)

    # Per-domain coverage + solved status + optional mean pLDDT
    enriched_domains: list[Domain] = []
    for domain in entry.domains:
        d_tuple = domain.range.as_tuple()
        d_cov = compute.domain_coverage_fraction(d_tuple, covered_ranges)
        d_solved = compute.is_solved(d_tuple, covered_ranges)

        d_plddt: float | None = None
        if plddt:
            start_i = d_tuple[0] - 1  # 1-indexed → 0-indexed
            end_i = d_tuple[1]         # exclusive slice end
            slice_ = plddt[start_i:end_i]
            d_plddt = sum(slice_) / len(slice_) if slice_ else None

        enriched_domains.append(
            domain.model_copy(
                update={
                    "coverage_fraction": round(d_cov, 4),
                    "solved": d_solved,
                    "mean_plddt": round(d_plddt, 2) if d_plddt is not None else None,
                }
            )
        )

    n_domains = len(enriched_domains)
    n_solved = sum(1 for d in enriched_domains if d.solved)
    solvable_domain_frac = (n_solved / n_domains) if n_domains > 0 else 0.0

    disorder_frac = compute.disordered_fraction(disordered, seq_len)

    # Mean pLDDT over uncovered *ordered* residues — used as confidence fallback
    # when no structures with resolution data are available.
    mean_plddt_uncovered: float | None = None
    if plddt:
        uncovered = compute.subtract_ranges((1, seq_len), covered_ranges)
        ordered_uncovered: list[tuple[int, int]] = []
        for region in uncovered:
            ordered_uncovered.extend(compute.subtract_ranges(region, disordered))
        if ordered_uncovered:
            vals: list[float] = []
            for start, end in ordered_uncovered:
                vals.extend(plddt[start - 1 : end])
            mean_plddt_uncovered = sum(vals) / len(vals) if vals else None

    # Confidence score (0..100):
    # Primary: fraction of structures with resolution < HIGH_RES_THRESHOLD_A, ×100.
    # Fallback: mean pLDDT over uncovered ordered residues (for uncharacterised proteins).
    res_values = [s.resolution_a for s in structures if s.resolution_a is not None]
    if res_values:
        n_high_res = sum(1 for r in res_values if r < scoring.HIGH_RES_THRESHOLD_A)
        high_res_fraction: float | None = n_high_res / len(res_values)
        confidence_score: float | None = high_res_fraction * 100.0
    else:
        high_res_fraction = None
        confidence_score = mean_plddt_uncovered  # pLDDT fallback

    # Missing regions: uncovered sub-spans of annotated domains + disordered.
    # min_length=10 trims trivial boundary gaps.
    domain_tuples = [d.range.as_tuple() for d in entry.domains]
    annotated = domain_tuples + disordered
    missing_tuples = compute.missing_regions(annotated, covered_ranges, min_length=10)
    missing_regions = [ResidueRange(start=s, end=e) for s, e in missing_tuples]

    return {
        "coverage_fraction": round(cov_frac, 4),
        "solvable_domain_fraction": round(solvable_domain_frac, 4),
        "disordered_fraction": round(disorder_frac, 4),
        "mean_plddt_uncovered": (
            round(mean_plddt_uncovered, 2) if mean_plddt_uncovered is not None else None
        ),
        "high_res_fraction": (
            round(high_res_fraction, 4) if high_res_fraction is not None else None
        ),
        "confidence_score": (
            round(confidence_score, 2) if confidence_score is not None else None
        ),
        "enriched_domains": enriched_domains,
        "missing_regions": missing_regions,
    }


# ── Phase 3b: Purification extraction ────────────────────────────────────────

def extract_purification_protocols(
    papers: list[dict],
    client: anthropic.Anthropic,
) -> list[PurificationProtocol]:
    """Extract structured purification protocols from PubMed abstracts.

    The LLM fills controlled-vocabulary fields (expression_system, steps, etc.)
    validated against the schema.  The ``notes`` field is the only free-text
    LLM output and must be grounded strictly in the abstract.
    """
    if not papers:
        return []

    papers_text = "\n\n---\n\n".join(
        f"PDB: {p['pdb_id']}\nTitle: {p['title']}\nPubMed ID: {p.get('pubmed_id', 'N/A')}\n\n"
        f"Abstract:\n{p['abstract']}"
        for p in papers
    )

    response = client.messages.parse(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_PURIFICATION_EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": papers_text}],
        output_format=_PurificationExtractionBatch,
    )

    if response.parsed_output is None:
        return []

    now = datetime.now(timezone.utc)
    paper_by_pdb = {p["pdb_id"]: p for p in papers}
    protocols: list[PurificationProtocol] = []

    for ext in response.parsed_output.protocols:
        paper = paper_by_pdb.get(ext.pdb_id)
        pmid = paper["pubmed_id"] if paper else None
        protocols.append(
            PurificationProtocol(
                pdb_id=ext.pdb_id,
                pubmed_id=pmid,
                expression_system=ext.expression_system,
                purification_steps=ext.purification_steps,
                requires_coexpression=ext.requires_coexpression,
                yield_category=ext.yield_category,
                construct_description=ext.construct_description,
                notes=ext.notes,
                provenance=Provenance(
                    source=Source.PUBMED,
                    identifier=pmid or ext.pdb_id,
                    url=(
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        if pmid else None
                    ),
                    retrieved_at=now,
                ),
            )
        )

    return protocols


# ── Phase 4: LLM narrative ────────────────────────────────────────────────────

class _Narrative(BaseModel):
    """The only LLM-authored fields in a TractabilityReport."""

    reasoning: list[str]
    recommended_strategy: list[str]


_NARRATIVE_SYSTEM = """\
You are writing the interpretation section of a protein structure tractability report.
A fact sheet with all computed values is provided.

Your output:
  reasoning            — 3–6 bullet strings explaining WHY the score is what it is.
                         Reference the computed facts (domains solved, disorder fraction,
                         high-res structure fraction, expression system, purification
                         steps, etc.).  Do not introduce any new numbers.
  recommended_strategy — 2–4 bullet strings suggesting concrete experimental steps
                         appropriate to this protein's tractability profile
                         (e.g. domain construct design, cryo-EM for complexes,
                         NMR for small disordered regions, co-expression partners,
                         preferred expression system based on prior success).

Hard rules:
- Write action-oriented bullets (not "I recommend X" but "Express the kinase domain
  independently (residues 100–350)").
- Every factual claim must be derivable from the fact sheet — do not hallucinate
  domain names, PDB IDs, expression systems, or organism details not present there.
- Do not produce any numbers that are not already in the fact sheet.
- Tone: expert structural biologist writing for a colleague.  Precise, not verbose.
"""


def _build_fact_sheet(
    query: str,
    accession: str,
    protein_name: str,
    organism: str,
    seq_len: int,
    facts: dict[str, Any],
    protocols: list[PurificationProtocol],
    purification_score: float | None,
    score: ScoreBreakdown,
    qc_flags: list[QCFlag],
) -> str:
    domain_lines = "\n".join(
        "  - {name} ({start}–{end}): {status}, coverage {cov:.0%}{plddt}".format(
            name=d.name,
            start=d.range.start,
            end=d.range.end,
            status="SOLVED" if d.solved else "UNSOLVED",
            cov=d.coverage_fraction,
            plddt=f", mean pLDDT {d.mean_plddt:.0f}" if d.mean_plddt is not None else "",
        )
        for d in facts["enriched_domains"]
    ) or "  (none annotated)"

    missing_str = (
        ", ".join(f"{r.start}–{r.end}" for r in facts["missing_regions"])
        or "none"
    )

    flag_lines = "\n".join(
        f"  [{f.severity.value.upper()}] {f.code}: {f.message}" for f in qc_flags
    ) or "  (none)"

    if facts["high_res_fraction"] is not None:
        conf_str = (
            f"High-res structures (< {scoring.HIGH_RES_THRESHOLD_A} Å): "
            f"{facts['high_res_fraction']:.1%} → confidence score {facts['confidence_score']:.1f} / 100"
        )
    elif facts["mean_plddt_uncovered"] is not None:
        conf_str = (
            f"Mean pLDDT over uncovered ordered residues: {facts['mean_plddt_uncovered']:.1f} "
            f"(used as confidence score; no structures with resolution data)"
        )
    else:
        conf_str = "N/A (no structures and no AlphaFold model)"

    if protocols:
        protocol_lines = "\n".join(
            "  {pdb}: {sys}, steps: {steps}{coexp}, yield: {yld}\n"
            "    Construct: {construct}\n"
            "    Notes: {notes}".format(
                pdb=p.pdb_id,
                sys=p.expression_system.value,
                steps=" → ".join(p.purification_steps) if p.purification_steps else "(unspecified)",
                coexp=", requires co-expression" if p.requires_coexpression else "",
                yld=p.yield_category,
                construct=p.construct_description or "(unspecified)",
                notes=p.notes,
            )
            for p in protocols
        )
        purif_str = (
            f"  Protocols from primary citations: {len(protocols)}\n"
            f"  Purification score: {purification_score:.1f} / 100\n"
            f"{protocol_lines}"
        )
    else:
        purif_str = "  No purification protocol data available (no high-res structures with PubMed citations)"

    return (
        f"Protein: {protein_name} ({accession}, {organism})\n"
        f"Query: {query}\n"
        f"Sequence length: {seq_len} aa\n"
        f"\n"
        f"Coverage:\n"
        f"  Experimental coverage fraction : {facts['coverage_fraction']:.1%}\n"
        f"  Disordered fraction            : {facts['disordered_fraction']:.1%}\n"
        f"  Confidence: {conf_str}\n"
        f"\n"
        f"Domains:\n{domain_lines}\n"
        f"\n"
        f"Missing regions (≥10 aa, in annotated domains/disordered spans):\n"
        f"  {missing_str}\n"
        f"\n"
        f"Purification tractability:\n{purif_str}\n"
        f"\n"
        f"Tractability score:\n"
        f"  Coverage points     : {score.coverage_points:.2f} / {scoring.COVERAGE_WEIGHT:.1f}\n"
        f"  Domain points       : {score.domain_points:.2f} / {scoring.DOMAIN_WEIGHT:.1f}\n"
        f"  Confidence points   : {score.confidence_points:.2f} / {scoring.CONFIDENCE_WEIGHT:.1f}\n"
        f"  Purification points : {score.purification_points:.2f} / {scoring.PURIFICATION_WEIGHT:.1f}\n"
        f"  Disorder penalty    : {score.disorder_penalty:.2f} / -{scoring.DISORDER_WEIGHT:.1f}\n"
        f"  TOTAL               : {score.total:.2f} / 100.0  [{score.rubric_version}]\n"
        f"\n"
        f"QC flags:\n{flag_lines}\n"
    )


def write_narrative(
    *,
    query: str,
    accession: str,
    protein_name: str,
    organism: str,
    seq_len: int,
    facts: dict[str, Any],
    protocols: list[PurificationProtocol],
    purification_score: float | None,
    score: ScoreBreakdown,
    qc_flags: list[QCFlag],
    client: anthropic.Anthropic,
) -> _Narrative:
    """Generate reasoning and recommended_strategy grounded in computed facts.

    Uses structured output so the response is validated against ``_Narrative``
    before it reaches the report.
    """
    fact_sheet = _build_fact_sheet(
        query=query,
        accession=accession,
        protein_name=protein_name,
        organism=organism,
        seq_len=seq_len,
        facts=facts,
        protocols=protocols,
        purification_score=purification_score,
        score=score,
        qc_flags=qc_flags,
    )

    response = client.messages.parse(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": fact_sheet}],
        output_format=_Narrative,
    )

    if response.parsed_output is None:
        raise ValueError("LLM did not return a parseable narrative response.")
    return response.parsed_output


# ── Public entry point ────────────────────────────────────────────────────────

def assess(query: str, api_key: str | None = None) -> TractabilityReport:
    """Assess the structural tractability of a protein.

    Args:
        query:   Free-text protein name, gene symbol, or bare UniProt accession.
        api_key: Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A fully populated ``TractabilityReport``.  Every numeric field is
        computed deterministically; ``reasoning`` and ``recommended_strategy``
        are the only LLM-authored fields, grounded in those numbers.
    """
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )

    # 1. Resolve the protein name to a UniProt accession.
    accession, _notes = resolve_protein(query, client)

    # 2. Acquire all data (no LLM).
    data = acquire_data(accession)
    entry: UniProtEntry = data["entry"]
    structures: list[ExperimentalStructure] = data["structures"]
    covered_ranges: list[tuple[int, int]] = data["covered_ranges"]
    plddt: list[float] | None = data["plddt"]
    ncbi: dict | None = data["ncbi"]

    # 3. Deterministic geometry.
    facts = compute_facts(entry, structures, covered_ranges, plddt)

    # 3b. Purification protocol extraction from high-res structure primary citations.
    high_res_pdb_ids = [
        s.pdb_id
        for s in structures
        if s.resolution_a is not None and s.resolution_a < scoring.HIGH_RES_THRESHOLD_A
    ]
    papers: list[dict] = []
    protocols: list[PurificationProtocol] = []
    purification_score_val: float | None = None

    if high_res_pdb_ids:
        try:
            papers = get_pubmed_abstracts(high_res_pdb_ids, max_papers=5)
            if papers:
                protocols = extract_purification_protocols(papers, client)
                purification_score_val = purification_tractability_score(protocols)
        except Exception as exc:
            print(f"[purification] warning: {exc}")

    # 4. Transparent additive score.
    score = scoring.score(
        coverage_fraction=facts["coverage_fraction"],
        solvable_domain_fraction=facts["solvable_domain_fraction"],
        confidence_score=facts["confidence_score"],
        purification_score=purification_score_val,
        disordered_fraction=facts["disordered_fraction"],
    )

    # 5. Cross-source QC.
    ncbi_length: int | None = ncbi.get("length") if isinstance(ncbi, dict) else None
    qc_flags = qc.run_all(
        uniprot_length=entry.sequence_length,
        ncbi_length=ncbi_length,
        covered_ranges=covered_ranges,
        disordered_ranges=list(entry.disordered_regions),
        structures=structures,
        domains=facts["enriched_domains"],
        plddt=plddt,
        coverage_fraction=facts["coverage_fraction"],
        disordered_fraction=facts["disordered_fraction"],
    )

    # 6. LLM narrative — grounded in computed facts only.
    narrative = write_narrative(
        query=query,
        accession=accession,
        protein_name=entry.protein_name,
        organism=entry.organism,
        seq_len=entry.sequence_length,
        facts=facts,
        protocols=protocols,
        purification_score=purification_score_val,
        score=score,
        qc_flags=qc_flags,
        client=client,
    )

    # 7. Assemble provenance list.
    all_provenance: list[Provenance] = [entry.provenance] + [
        s.provenance for s in structures
    ] + [
        p.provenance for p in protocols
    ]

    return TractabilityReport(
        query=query,
        resolved_accession=accession,
        protein_name=entry.protein_name,
        organism=entry.organism,
        sequence_length=entry.sequence_length,
        experimental_structures=structures,
        overall_coverage_fraction=facts["coverage_fraction"],
        domains=facts["enriched_domains"],
        missing_regions=facts["missing_regions"],
        disordered_fraction=facts["disordered_fraction"],
        mean_plddt_uncovered=facts["mean_plddt_uncovered"],
        purification_protocols=protocols,
        purification_score=purification_score_val,
        high_res_fraction=facts["high_res_fraction"],
        confidence_score=facts["confidence_score"],
        score=score,
        reasoning=narrative.reasoning,
        recommended_strategy=narrative.recommended_strategy,
        qc_flags=qc_flags,
        provenance=all_provenance,
        generated_at=datetime.now(timezone.utc),
    )
