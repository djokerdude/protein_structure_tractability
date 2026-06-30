"""Streamlit web UI for the tractable pipeline.

Run with:
    streamlit run src/tractable/ui.py
or, after `pip install -e .`:
    tractable-ui
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import streamlit as st

from tractable import qc, scoring
from tractable.agent import (
    acquire_data,
    compute_facts,
    extract_purification_protocols,
    write_narrative,
)
from tractable.purification import purification_tractability_score
from tractable.schema import Provenance, TractabilityReport
from tractable.tools import get_pubmed_abstracts, search_uniprot

_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
    re.IGNORECASE,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="tractable",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    _api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Leave blank to use the ANTHROPIC_API_KEY environment variable.",
    )
    st.divider()
    st.caption(
        "**tractable** assesses how amenable a protein is to experimental "
        "structure determination.\n\n"
        "Data from UniProt · RCSB PDB · AlphaFold · NCBI · PubMed Central."
    )


def _api_key() -> str:
    return _api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")


# ── Session state helpers ─────────────────────────────────────────────────────

def _reset() -> None:
    for key in ("phase", "query", "candidates", "accession", "report", "error"):
        st.session_state.pop(key, None)


if "phase" not in st.session_state:
    st.session_state.phase = "search"

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## tractable")
st.caption("Protein structure tractability assessment")

# ── Phase: search ─────────────────────────────────────────────────────────────

if st.session_state.phase == "search":
    if "error" in st.session_state:
        st.error(st.session_state.pop("error"))

    with st.form("search_form"):
        query = st.text_input(
            "Protein",
            placeholder="Gene symbol, protein name, or UniProt accession (e.g. SARM1, PIK3CA, P42336)",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Assess →", type="primary", use_container_width=True)

    if submitted and query.strip():
        q = query.strip()
        st.session_state.query = q

        if _ACCESSION_RE.match(q):
            # Bare accession — skip search
            st.session_state.accession = q.upper()
            st.session_state.phase = "running"
        else:
            with st.spinner("Searching UniProt…"):
                try:
                    candidates = search_uniprot(q, limit=10)
                except Exception as exc:
                    st.error(f"UniProt search failed: {exc}")
                    candidates = []

            if not candidates:
                st.error(f"No UniProt entries found for {q!r}.")
            elif len(candidates) == 1:
                st.session_state.accession = candidates[0].accession
                st.session_state.phase = "running"
            else:
                # Auto-select if there is exactly one reviewed Homo sapiens entry
                human_reviewed = [
                    c for c in candidates
                    if c.reviewed and "homo sapiens" in c.organism.lower()
                ]
                if len(human_reviewed) == 1:
                    st.session_state.accession = human_reviewed[0].accession
                    st.session_state.phase = "running"
                else:
                    st.session_state.candidates = candidates
                    st.session_state.phase = "disambiguate"

        st.rerun()

# ── Phase: disambiguate ───────────────────────────────────────────────────────

elif st.session_state.phase == "disambiguate":
    candidates = st.session_state.candidates
    q = st.session_state.query

    st.markdown(f"**{len(candidates)} candidates for \"{q}\"** — select the target:")
    st.caption("★ = reviewed (Swiss-Prot curated)")

    options = [
        "{acc}{star} — {name} · {org} · {length:,} aa".format(
            acc=c.accession,
            star=" ★" if c.reviewed else "",
            name=c.protein_name,
            org=c.organism,
            length=c.sequence_length,
        )
        for c in candidates
    ]

    selection = st.radio("Candidates", options, label_visibility="collapsed")

    col1, col2 = st.columns([3, 1])
    if col1.button("Run assessment →", type="primary", use_container_width=True):
        idx = options.index(selection)
        st.session_state.accession = candidates[idx].accession
        st.session_state.phase = "running"
        st.rerun()
    if col2.button("← Back", use_container_width=True):
        _reset()
        st.rerun()

# ── Phase: running ────────────────────────────────────────────────────────────

elif st.session_state.phase == "running" and "report" not in st.session_state:
    accession = st.session_state.accession
    q = st.session_state.query
    api_key = _api_key()

    if not api_key:
        st.error(
            "No Anthropic API key found. "
            "Set the **ANTHROPIC_API_KEY** environment variable or enter it in the sidebar."
        )
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    error_msg: str | None = None

    with st.status(f"Assessing **{accession}**…", expanded=True) as status:
        try:
            st.write("Fetching UniProt entry and experimental structures…")
            data = acquire_data(accession)
            entry = data["entry"]
            structures = data["structures"]
            covered_ranges = data["covered_ranges"]
            plddt = data["plddt"]
            ncbi = data["ncbi"]

            st.write(
                f"Computing coverage for **{entry.protein_name}** "
                f"({entry.sequence_length:,} aa, {len(structures)} structures)…"
            )
            facts = compute_facts(entry, structures, covered_ranges, plddt)

            high_res_ids = [
                s.pdb_id
                for s in structures
                if s.resolution_a is not None
                and s.resolution_a < scoring.HIGH_RES_THRESHOLD_A
            ]
            papers: list[dict] = []
            protocols = []
            purif_score: float | None = None

            if high_res_ids:
                st.write(
                    f"Searching PubMed for purification protocols "
                    f"({len(high_res_ids)} high-res structures)…"
                )
                try:
                    papers = get_pubmed_abstracts(high_res_ids, max_papers=5)
                    if papers:
                        st.write(
                            f"Extracting purification protocols from "
                            f"{len(papers)} paper(s) (LLM)…"
                        )
                        protocols = extract_purification_protocols(papers, client)
                        purif_score = purification_tractability_score(protocols)
                except Exception as exc:
                    st.warning(f"Purification extraction skipped: {exc}")

            st.write("Computing score and running QC checks…")
            score = scoring.score(
                coverage_fraction=facts["coverage_fraction"],
                solvable_domain_fraction=facts["solvable_domain_fraction"],
                confidence_score=facts["confidence_score"],
                purification_score=purif_score,
                disordered_fraction=facts["disordered_fraction"],
            )
            ncbi_length = ncbi.get("length") if isinstance(ncbi, dict) else None
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

            st.write("Writing reasoning and strategy (LLM)…")
            narrative = write_narrative(
                query=q,
                accession=accession,
                protein_name=entry.protein_name,
                organism=entry.organism,
                seq_len=entry.sequence_length,
                facts=facts,
                protocols=protocols,
                purification_score=purif_score,
                score=score,
                qc_flags=qc_flags,
                client=client,
            )

            all_provenance = (
                [entry.provenance]
                + [s.provenance for s in structures]
                + [p.provenance for p in protocols]
            )
            report = TractabilityReport(
                query=q,
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
                purification_score=purif_score,
                high_res_fraction=facts["high_res_fraction"],
                confidence_score=facts["confidence_score"],
                score=score,
                reasoning=narrative.reasoning,
                recommended_strategy=narrative.recommended_strategy,
                qc_flags=qc_flags,
                provenance=all_provenance,
                generated_at=datetime.now(timezone.utc),
            )
            st.session_state.report = report
            status.update(label="Assessment complete!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Assessment failed", state="error", expanded=False)
            error_msg = str(exc)

    if error_msg:
        st.session_state.error = error_msg
        st.session_state.phase = "search"
    else:
        st.session_state.phase = "report"

    st.rerun()

# ── Phase: report ─────────────────────────────────────────────────────────────

elif st.session_state.phase == "report" and "report" in st.session_state:
    report: TractabilityReport = st.session_state.report
    sc = report.score

    # ── Header row ────────────────────────────────────────────────────────────
    col_head, col_score = st.columns([4, 1])
    with col_head:
        st.subheader(report.protein_name)
        n_structs = len(report.experimental_structures)
        st.caption(
            f"**{report.resolved_accession}** · {report.organism} · "
            f"{report.sequence_length:,} aa · {n_structs} structure{'s' if n_structs != 1 else ''}"
        )
    with col_score:
        st.metric(
            "Tractability score",
            f"{sc.total:.1f} / 100",
            help=f"Rubric: {sc.rubric_version}",
        )

    # ── Score breakdown ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Coverage",
        f"{sc.coverage_points:.1f} / 35",
        help=f"{report.overall_coverage_fraction:.1%} of sequence covered",
    )
    n_solved = sum(1 for d in report.domains if d.solved)
    n_dom = len(report.domains)
    c2.metric(
        "Domains",
        f"{sc.domain_points:.1f} / 25",
        help=f"{n_solved}/{n_dom} annotated domains solved",
    )
    hi_frac = report.high_res_fraction or 0.0
    c3.metric(
        "Confidence",
        f"{sc.confidence_points:.1f} / 20",
        help=f"{hi_frac:.1%} of structures < {scoring.HIGH_RES_THRESHOLD_A} Å",
    )
    c4.metric(
        "Purification",
        f"{sc.purification_points:.1f} / 20",
        help=(
            f"Score: {report.purification_score:.1f}/100"
            if report.purification_score is not None
            else "No protocol data"
        ),
    )
    c5.metric(
        "Disorder",
        f"{sc.disorder_penalty:.1f} / −20",
        delta=f"{report.disordered_fraction:.1%} disordered" if report.disordered_fraction else None,
        delta_color="inverse" if report.disordered_fraction > 0.05 else "off",
    )

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_overview, tab_structures, tab_purification, tab_json = st.tabs(
        ["Overview", "Structures", "Purification", "Raw JSON"]
    )

    # ── Overview tab ──────────────────────────────────────────────────────────
    with tab_overview:
        col_r, col_s = st.columns(2)

        with col_r:
            st.subheader("Reasoning")
            for bullet in report.reasoning:
                st.markdown(f"- {bullet}")

        with col_s:
            st.subheader("Recommended Strategy")
            for bullet in report.recommended_strategy:
                st.markdown(f"- {bullet}")

        st.subheader("Domains")
        if report.domains:
            st.dataframe(
                [
                    {
                        "Domain": d.name,
                        "Range": f"{d.range.start}–{d.range.end}",
                        "Coverage": f"{d.coverage_fraction:.0%}",
                        "Solved": "✓" if d.solved else "✗",
                        "pLDDT": f"{d.mean_plddt:.0f}" if d.mean_plddt is not None else "—",
                    }
                    for d in report.domains
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No annotated domains.")

        if report.missing_regions:
            st.subheader("Missing Regions")
            st.write(
                "Uncovered spans ≥ 10 aa in annotated domains or disordered regions: "
                + ", ".join(f"{r.start}–{r.end}" for r in report.missing_regions)
            )

        if report.qc_flags:
            st.subheader("QC Flags")
            for flag in report.qc_flags:
                fn = {
                    "error": st.error,
                    "warning": st.warning,
                    "info": st.info,
                }.get(flag.severity.value, st.info)
                fn(f"**{flag.code}**: {flag.message}")

    # ── Structures tab ────────────────────────────────────────────────────────
    with tab_structures:
        methods = sorted({s.method for s in report.experimental_structures})
        selected_methods = st.multiselect(
            "Filter by method",
            options=methods,
            default=methods,
        )
        filtered = [
            s for s in report.experimental_structures if s.method in selected_methods
        ]
        st.caption(f"Showing {len(filtered)} of {len(report.experimental_structures)} structures")
        st.dataframe(
            [
                {
                    "PDB ID": s.pdb_id,
                    "Method": s.method,
                    "Resolution (Å)": s.resolution_a,
                    "Range": f"{s.covered_range.start}–{s.covered_range.end}",
                }
                for s in filtered
            ],
            hide_index=True,
            use_container_width=True,
        )

    # ── Purification tab ──────────────────────────────────────────────────────
    with tab_purification:
        if report.purification_protocols:
            n_total = len(report.purification_protocols)
            n_methods = sum(
                1 for p in report.purification_protocols if p.text_source == "methods"
            )
            n_paywall = n_total - n_methods
            if n_paywall:
                st.info(
                    f"{n_methods}/{n_total} papers with full Methods section · "
                    f"{n_paywall} abstract only "
                    f"({n_paywall / n_total:.0%} behind paywall — "
                    f"only publicly available papers were searched)."
                )
            else:
                st.success(f"All {n_total} paper(s) have open-access Methods sections.")

            for p in report.purification_protocols:
                with st.expander(
                    f"**{p.pdb_id}** · PMID {p.pubmed_id or 'N/A'} · [{p.text_source}]"
                ):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Expression System", p.expression_system.value)
                    m2.metric("Yield", p.yield_category)
                    m3.metric(
                        "Co-expression Required",
                        "Yes" if p.requires_coexpression else "No",
                    )
                    if p.purification_steps:
                        st.write("**Steps:** " + " → ".join(p.purification_steps))
                    else:
                        st.write("**Steps:** (unspecified)")
                    st.write("**Construct:** " + (p.construct_description or "(unspecified)"))
                    st.write("**Notes:** " + p.notes)
        else:
            st.info(
                "No purification protocols available — "
                "no high-res structures with PubMed citations were found."
            )

    # ── Raw JSON tab ──────────────────────────────────────────────────────────
    with tab_json:
        st.json(report.model_dump(mode="json"), expanded=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("← New assessment"):
        _reset()
        st.rerun()


# ── Console entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Launch the Streamlit app (used by the `tractable-ui` console script)."""
    import subprocess
    import sys

    sys.exit(
        subprocess.call(
            ["streamlit", "run", str(Path(__file__).resolve())]
        )
    )
