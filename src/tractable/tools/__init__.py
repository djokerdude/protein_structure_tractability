"""Data-source tools exposed to the agent.

Each function is a single, well-typed unit of data acquisition. They are the
*tools* the LLM agent is allowed to call; the agent decides which to invoke and
in what order, but the functions themselves are deterministic, cached, and
provenance-stamped. Keeping them small and pure is what makes the agent
auditable and the pipeline reproducible.

Real endpoints:

* UniProt REST .............. https://rest.uniprot.org/uniprotkb/search
* RCSB PDB Search API ........ https://search.rcsb.org/rcsbsearch/v2/query
* RCSB PDB Data API .......... https://data.rcsb.org/graphql
  (polymer_entities query provides SIFTS-derived UniProt alignment ranges;
   the legacy PDBe /pdbe/api/mappings/ endpoint is defunct as of 2024)
* AlphaFold DB .............. https://alphafold.ebi.ac.uk/api/prediction/
* NCBI E-utilities .......... https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Cache: tests/fixtures/ — one JSON file per request, keyed by a readable name.
Cached responses mean tests run offline and we respect each service's rate
limits. Commit representative fixture files alongside tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx

from ..schema import (
    Candidate,
    Domain,
    ExperimentalStructure,
    Provenance,
    ResidueRange,
    Source,
)

Range = tuple[int, int]

# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
_TIMEOUT = 30.0


def _cpath(prefix: str, key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80]
    return _CACHE_DIR / f"{prefix}__{safe}.json"


def _cload(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _csave(path: Path, data: dict | list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── UniProtEntry protocol + concrete implementation ───────────────────────────


class UniProtEntry(Protocol):
    """Shape of a resolved UniProt record (sequence + annotations)."""

    accession: str
    protein_name: str
    organism: str
    sequence: str
    sequence_length: int
    domains: list[Domain]
    disordered_regions: list[Range]
    provenance: Provenance


@dataclass
class _UniProtEntryImpl:
    accession: str
    protein_name: str
    organism: str
    sequence: str
    sequence_length: int
    domains: list[Domain]
    disordered_regions: list[Range]
    provenance: Provenance


# ── search_uniprot ────────────────────────────────────────────────────────────


def search_uniprot(query: str, limit: int = 10) -> list[Candidate]:
    """Resolve a free-text protein name to candidate UniProt accessions.

    This is the entity-resolution step. The agent reasons over the returned
    candidates (organism, reviewed vs. unreviewed, isoform) to pick one, or
    asks the user to disambiguate when it is genuinely ambiguous.
    """
    cache = _cpath("uniprot_search", f"{query}_{limit}")
    raw = _cload(cache)

    if raw is None:
        resp = httpx.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": query,
                "format": "json",
                "size": limit,
                "fields": "accession,protein_name,gene_names,organism_name,length",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
        _csave(cache, raw)

    candidates: list[Candidate] = []
    for result in (raw or {}).get("results", []):
        acc = result["primaryAccession"]

        # protein name: prefer recommendedName, fall back to submittedName
        pd = result.get("proteinDescription", {})
        rn = pd.get("recommendedName", {})
        name = rn.get("fullName", {}).get("value", "")
        if not name:
            sn = pd.get("submittedName", [{}])
            name = sn[0].get("fullName", {}).get("value", acc) if sn else acc

        gene: str | None = None
        genes = result.get("genes", [])
        if genes:
            gene = genes[0].get("geneName", {}).get("value")

        organism = result.get("organism", {}).get("scientificName", "Unknown")
        seq_len = result.get("sequence", {}).get("length", 0)
        reviewed = "reviewed" in result.get("entryType", "").lower()

        candidates.append(
            Candidate(
                accession=acc,
                protein_name=name,
                gene=gene,
                organism=organism,
                sequence_length=seq_len,
                reviewed=reviewed,
            )
        )

    return candidates


# ── get_uniprot_entry ─────────────────────────────────────────────────────────


def get_uniprot_entry(accession: str) -> _UniProtEntryImpl:
    """Fetch sequence, length, domain features, and disordered regions.

    Domain ranges come from UniProt feature annotations (DOMAIN/REGION);
    disordered regions from UniProt 'Region: Disordered' features, optionally
    augmented by a low-pLDDT proxy from AlphaFold.
    """
    cache = _cpath("uniprot_entry", accession)
    raw = _cload(cache)

    if raw is None:
        resp = httpx.get(
            f"https://rest.uniprot.org/uniprotkb/{accession}",
            params={"format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
        _csave(cache, raw)

    seq_info = raw.get("sequence", {})
    sequence = seq_info.get("value", "")
    seq_len = seq_info.get("length", len(sequence))

    # Protein name
    pd = raw.get("proteinDescription", {})
    rn = pd.get("recommendedName", {})
    protein_name = rn.get("fullName", {}).get("value", "")
    if not protein_name:
        sn = pd.get("submittedName", [{}])
        protein_name = sn[0].get("fullName", {}).get("value", accession) if sn else accession

    organism = raw.get("organism", {}).get("scientificName", "Unknown")

    domains: list[Domain] = []
    disordered: list[Range] = []

    for feat in raw.get("features", []):
        ftype = feat.get("type", "")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None or end is None or start > end or start < 1:
            continue

        if ftype == "Domain":
            domain_name = feat.get("description", f"Domain {start}–{end}")
            domains.append(
                Domain(
                    name=domain_name,
                    range=ResidueRange(start=start, end=end),
                    source=Source.UNIPROT,
                    coverage_fraction=0.0,
                    solved=False,
                )
            )
        elif ftype == "Region" and "disorder" in feat.get("description", "").lower():
            disordered.append((start, end))

    return _UniProtEntryImpl(
        accession=accession,
        protein_name=protein_name,
        organism=organism,
        sequence=sequence,
        sequence_length=seq_len,
        domains=domains,
        disordered_regions=disordered,
        provenance=Provenance(
            source=Source.UNIPROT,
            identifier=accession,
            url=f"https://rest.uniprot.org/uniprotkb/{accession}",
            retrieved_at=_now(),
        ),
    )


# ── RCSB shared fetch (replaces defunct PDBe SIFTS REST endpoint) ─────────────
#
# The legacy PDBe /pdbe/api/mappings/uniprot/{acc} endpoint returns 404.
# RCSB's polymer_entities query exposes the same SIFTS-derived alignment data:
#   rcsb_polymer_entity_align.aligned_regions gives UniProt residue ranges
#   (ref_beg_seq_id + length) per polymer entity, which is the authoritative
#   residue-level coverage source.

_RCSB_ENTITY_GQL = """
query GetEntities($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    entry {
      rcsb_id
      exptl { method }
      rcsb_entry_info { resolution_combined }
    }
    rcsb_polymer_entity_align {
      reference_database_accession
      aligned_regions { ref_beg_seq_id length }
    }
  }
}
"""


def _fetch_rcsb_entities(accession: str) -> list[dict]:
    """Return one record per PDB entry covering this UniProt accession.

    Each record: {entry_id, method, resolution, unp_ranges: list[tuple[int,int]]}.
    Results are cached so both get_pdb_structures and get_sifts_coverage are
    served from a single network round-trip.
    """
    cache = _cpath("rcsb_entities", accession)
    cached = _cload(cache)
    if cached is not None:
        return cached  # type: ignore[return-value]

    # Step 1 — find all polymer entity IDs that reference this UniProt accession.
    search_resp = httpx.post(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        json={
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers"
                                 ".reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": accession,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {"return_all_hits": True},
        },
        timeout=_TIMEOUT,
    )
    search_resp.raise_for_status()
    entity_ids = [x["identifier"] for x in search_resp.json().get("result_set", [])]

    if not entity_ids:
        _csave(cache, [])
        return []

    # Step 2 — batch-fetch entity details + alignment regions (chunks of 50).
    by_entry: dict[str, dict] = {}
    for i in range(0, len(entity_ids), 50):
        chunk = entity_ids[i : i + 50]
        resp = httpx.post(
            "https://data.rcsb.org/graphql",
            json={"query": _RCSB_ENTITY_GQL, "variables": {"ids": chunk}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        for entity in resp.json().get("data", {}).get("polymer_entities", []) or []:
            entry = entity.get("entry", {})
            entry_id: str = entry.get("rcsb_id", "")
            if not entry_id:
                continue

            method = (entry.get("exptl") or [{}])[0].get("method", "UNKNOWN")
            res_list = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
            resolution: float | None = res_list[0] if res_list else None

            for align in entity.get("rcsb_polymer_entity_align") or []:
                if align.get("reference_database_accession") != accession:
                    continue
                for region in align.get("aligned_regions") or []:
                    beg = region.get("ref_beg_seq_id")
                    ln = region.get("length")
                    if not beg or not ln:
                        continue
                    rec = by_entry.setdefault(
                        entry_id,
                        {"entry_id": entry_id, "method": method,
                         "resolution": resolution, "unp_ranges": []},
                    )
                    rec["unp_ranges"].append([beg, beg + ln - 1])

    results = list(by_entry.values())
    _csave(cache, results)
    return results


# ── get_pdb_structures ────────────────────────────────────────────────────────


def get_pdb_structures(accession: str) -> list[ExperimentalStructure]:
    """Find experimental PDB structures for a UniProt accession.

    Uses RCSB search + polymer entity alignment data for residue-level coverage.
    """
    from .. import compute

    entities = _fetch_rcsb_entities(accession)
    now = _now()
    structures: list[ExperimentalStructure] = []

    for rec in entities:
        ranges = [tuple(r) for r in rec["unp_ranges"]]
        if not ranges:
            continue
        merged = compute.merge_ranges(ranges)  # type: ignore[arg-type]
        unp_start = merged[0][0]
        unp_end = merged[-1][1]
        entry_id: str = rec["entry_id"]
        structures.append(
            ExperimentalStructure(
                pdb_id=entry_id,
                method=rec["method"],
                resolution_a=rec["resolution"],
                covered_range=ResidueRange(start=unp_start, end=unp_end),
                provenance=Provenance(
                    source=Source.RCSB_PDB,
                    identifier=entry_id,
                    url=f"https://www.rcsb.org/structure/{entry_id}",
                    retrieved_at=now,
                ),
            )
        )

    return sorted(structures, key=lambda s: s.covered_range.start)


# ── get_sifts_coverage ────────────────────────────────────────────────────────


def get_sifts_coverage(accession: str) -> list[Range]:
    """Per-residue UniProt->PDB coverage ranges (SIFTS-derived, via RCSB).

    This is the authoritative source for *which residues* are experimentally
    covered, and therefore for the coverage percentage. Ranges are from
    rcsb_polymer_entity_align, which carries the same SIFTS residue mappings
    that the legacy PDBe endpoint served.
    """
    from .. import compute

    entities = _fetch_rcsb_entities(accession)
    ranges: list[Range] = []
    for rec in entities:
        for r in rec["unp_ranges"]:
            ranges.append((r[0], r[1]))
    return compute.merge_ranges(ranges)


# ── get_alphafold_plddt ───────────────────────────────────────────────────────


def get_alphafold_plddt(accession: str) -> list[float]:
    """Per-residue AlphaFold pLDDT (index i -> confidence for residue i+1).

    Fetches the AlphaFold PDB file and reads B-factor values from CA atoms.
    In AlphaFold models the B-factor column stores the pLDDT score (0–100).
    """
    # Step 1: fetch metadata to get the PDB file URL
    meta_cache = _cpath("alphafold_meta", accession)
    meta_raw = _cload(meta_cache)

    if meta_raw is None:
        resp = httpx.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        meta_raw = resp.json()
        _csave(meta_cache, meta_raw)

    if not meta_raw:
        raise ValueError(f"No AlphaFold prediction found for {accession}")

    pdb_url: str = meta_raw[0]["pdbUrl"]

    # Step 2: fetch pLDDT from the PDB file (CA B-factors)
    plddt_cache = _cpath("alphafold_plddt", accession)
    plddt_raw = _cload(plddt_cache)

    if plddt_raw is None:
        resp = httpx.get(pdb_url, timeout=60.0)
        resp.raise_for_status()
        plddt_vals: list[float] = []
        for line in resp.text.splitlines():
            # ATOM records: columns 13-16 = atom name, 61-66 = B-factor
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    plddt_vals.append(float(line[60:66]))
                except ValueError:
                    continue
        plddt_raw = plddt_vals
        _csave(plddt_cache, plddt_raw)

    return list(plddt_raw)  # type: ignore[arg-type]


# ── get_ncbi_record ───────────────────────────────────────────────────────────


def get_ncbi_record(accession_or_name: str) -> dict:
    """Fetch an NCBI protein record for cross-source QC (e.g. length check).

    Used by the QC layer to flag inconsistencies between databases rather than
    as a primary data source.
    """
    cache = _cpath("ncbi", accession_or_name)
    cached = _cload(cache)
    if cached is not None:
        return cached  # type: ignore[return-value]

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Try accession lookup first, then fall back to gene+organism search
    for term in (
        f"{accession_or_name}[accession]",
        f"{accession_or_name}[gene] AND human[organism] AND refseq[filter]",
    ):
        r = httpx.get(
            f"{base}/esearch.fcgi",
            params={"db": "protein", "term": term, "retmode": "json", "retmax": 5},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if ids:
            break
    else:
        result: dict = {"error": f"No NCBI protein record found for {accession_or_name!r}"}
        _csave(cache, result)
        return result

    uid = ids[0]
    r2 = httpx.get(
        f"{base}/esummary.fcgi",
        params={"db": "protein", "id": uid, "retmode": "json"},
        timeout=_TIMEOUT,
    )
    r2.raise_for_status()
    doc = r2.json().get("result", {}).get(uid, {})

    result = {
        "uid": uid,
        "accession": doc.get("accessionversion", ""),
        "title": doc.get("title", ""),
        "length": doc.get("slen"),
    }
    _csave(cache, result)
    return result


# ── get_pubmed_abstracts ──────────────────────────────────────────────────────

_RCSB_CITATION_GQL = """
query GetCitations($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_primary_citation {
      pdbx_database_id_PubMed
      title
    }
  }
}
"""


def get_pubmed_abstracts(pdb_ids: list[str], max_papers: int = 5) -> list[dict]:
    """Fetch PubMed abstracts for the primary citations of PDB structures.

    Uses RCSB GraphQL to resolve PubMed IDs, then NCBI efetch for the abstract
    text.  Only entries with a PubMed citation and a non-empty abstract are
    returned.  Both the citation lookup (per batch) and each abstract are cached
    individually so re-runs are offline.

    Returns a list of dicts: {pdb_id, pubmed_id, title, abstract}.
    """
    if not pdb_ids:
        return []

    target_ids = pdb_ids[:max_papers]

    # Step 1: resolve PubMed IDs via RCSB citation metadata (batch).
    batch_key = "_".join(sorted(target_ids))
    citations_cache = _cpath("rcsb_citations", batch_key)
    citations_raw = _cload(citations_cache)

    if citations_raw is None:
        resp = httpx.post(
            "https://data.rcsb.org/graphql",
            json={"query": _RCSB_CITATION_GQL, "variables": {"ids": target_ids}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        citations_raw = resp.json().get("data", {}).get("entries", []) or []
        _csave(citations_cache, citations_raw)

    pmid_map: dict[str, dict] = {}  # pmid → {pdb_id, title}
    for entry in citations_raw:
        pdb_id = entry.get("rcsb_id", "")
        cite = entry.get("rcsb_primary_citation") or {}
        pmid = cite.get("pdbx_database_id_PubMed")
        title = cite.get("title", "")
        if pmid and pdb_id:
            pmid_map[str(pmid)] = {"pdb_id": pdb_id, "title": title}

    # Step 2: fetch abstract text from NCBI for each PubMed ID.
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    papers: list[dict] = []

    for pmid, meta in pmid_map.items():
        abstract_cache = _cpath("pubmed_abstract", pmid)
        abstract_data = _cload(abstract_cache)

        if abstract_data is None:
            try:
                r = httpx.get(
                    f"{base}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": pmid,
                        "rettype": "abstract",
                        "retmode": "text",
                    },
                    timeout=_TIMEOUT,
                )
                r.raise_for_status()
                abstract_text = r.text
            except Exception:
                abstract_text = ""
            abstract_data = {"text": abstract_text}
            _csave(abstract_cache, abstract_data)

        abstract = abstract_data.get("text", "") if isinstance(abstract_data, dict) else ""
        if abstract.strip():
            papers.append(
                {
                    "pdb_id": meta["pdb_id"],
                    "pubmed_id": pmid,
                    "title": meta["title"],
                    "abstract": abstract,
                }
            )

    return papers
