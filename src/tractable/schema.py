"""Typed data model for the structure-determination tractability report.

This module is the contract for the whole pipeline. Every other module either
produces or consumes these models:

  tools/*       -> raw records (UniProtEntry, ExperimentalStructure, ...)
  compute       -> derived facts (coverage, solved domains, missing regions)
  purification  -> PurificationProtocol, purification_tractability_score
  scoring       -> ScoreBreakdown
  agent         -> reasoning + recommended_strategy (LLM, grounded in the above)

Design rule enforced by this schema: the LLM never invents numbers. Every
numeric field is populated by deterministic code in `compute`/`scoring`. The
LLM only fills `reasoning` and `recommended_strategy`, and only from the
already-computed facts it is handed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Where a datum came from. Used everywhere for provenance."""

    UNIPROT = "uniprot"
    RCSB_PDB = "rcsb_pdb"
    PDBE_SIFTS = "pdbe_sifts"
    ALPHAFOLD = "alphafold"
    NCBI = "ncbi"
    PUBMED = "pubmed"


class Provenance(BaseModel):
    """Traceability record attached to anything we report.

    Lets a reader reproduce or audit any field: which database, which
    identifier, fetched when. Maps directly to the role's requirement to
    "document data provenance ... to support reproducibility and auditing".
    """

    source: Source
    identifier: str  # accession / PDB ID / raw query string
    url: Optional[str] = None
    retrieved_at: datetime


class ResidueRange(BaseModel):
    """A 1-indexed, inclusive residue span (UniProt numbering)."""

    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


class Candidate(BaseModel):
    """A possible match during entity resolution (name -> accession).

    The agent reasons over a list of these to disambiguate organism/isoform
    before any heavy data acquisition happens.
    """

    accession: str
    protein_name: str
    gene: Optional[str] = None
    organism: str
    sequence_length: int
    reviewed: bool  # UniProtKB/Swiss-Prot (curated) vs TrEMBL


class Domain(BaseModel):
    """An annotated structural/functional domain and its solved status."""

    name: str
    range: ResidueRange
    source: Source
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    solved: bool  # coverage_fraction >= SOLVED_THRESHOLD (see compute)
    mean_plddt: Optional[float] = None


class ExperimentalStructure(BaseModel):
    """One experimentally determined structure mapped onto the sequence."""

    pdb_id: str
    method: str  # "X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR", ...
    resolution_a: Optional[float] = None
    covered_range: ResidueRange
    provenance: Provenance


class QCSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QCFlag(BaseModel):
    """An anomaly or inconsistency detected during cross-source validation.

    Maps to the role's "detect anomalies, flag inconsistencies, and enforce
    data standards".
    """

    severity: QCSeverity
    code: str  # machine-readable, e.g. "LENGTH_MISMATCH", "NO_STRUCTURES_FOUND"
    message: str
    sources_involved: list[Source]


# ── Purification ──────────────────────────────────────────────────────────────


class ExpressionSystem(str, Enum):
    """Host system used to produce the protein for structural studies."""

    ECOLI = "ecoli"
    INSECT = "insect"        # Sf9, Hi5, baculovirus
    MAMMALIAN = "mammalian"  # HEK293, CHO, COS
    YEAST = "yeast"          # S. cerevisiae, P. pastoris
    CELL_FREE = "cell_free"
    UNKNOWN = "unknown"


YieldCategory = Literal["high", "medium", "low", "unknown"]


class PurificationProtocol(BaseModel):
    """LLM-extracted purification protocol from one PDB primary citation.

    The `notes` field is the only LLM-authored field; all other fields are
    structured extractions validated against controlled vocabularies.
    """

    pdb_id: str
    pubmed_id: Optional[str] = None
    expression_system: ExpressionSystem
    purification_steps: list[str]  # e.g. ["Ni-NTA affinity", "ion exchange", "SEC"]
    requires_coexpression: bool = False
    yield_category: YieldCategory = "unknown"
    construct_description: str  # e.g. "TIR domain (560–724), N-terminal His6"
    notes: str  # one-line LLM summary grounded strictly in the abstract
    provenance: Provenance


# ── Scoring ───────────────────────────────────────────────────────────────────


class ScoreBreakdown(BaseModel):
    """Transparent, additive score. Sub-scores make the total auditable."""

    coverage_points: float
    domain_points: float
    confidence_points: float
    purification_points: float
    disorder_penalty: float  # <= 0
    total: float = Field(ge=0.0, le=100.0)
    rubric_version: str


# ── Report ────────────────────────────────────────────────────────────────────


class TractabilityReport(BaseModel):
    """The full deliverable for one protein."""

    # --- identity (resolved) ---
    query: str
    resolved_accession: str
    protein_name: str
    organism: str
    sequence_length: int

    # --- evidence (deterministic) ---
    experimental_structures: list[ExperimentalStructure]
    overall_coverage_fraction: float = Field(ge=0.0, le=1.0)
    domains: list[Domain]
    missing_regions: list[ResidueRange]
    disordered_fraction: float = Field(ge=0.0, le=1.0)
    mean_plddt_uncovered: Optional[float] = None  # pLDDT fallback; None when res-based confidence used

    # --- purification (LLM-extracted, deterministically scored) ---
    purification_protocols: list[PurificationProtocol] = []
    purification_score: Optional[float] = None  # 0..100 aggregate purification tractability

    # --- assessment (deterministic score, LLM narrative) ---
    high_res_fraction: Optional[float] = None  # fraction of structures < HIGH_RES_THRESHOLD_A
    confidence_score: Optional[float] = None   # 0..100; drives confidence_points in score
    score: ScoreBreakdown
    reasoning: list[str]  # LLM, grounded strictly in the fields above
    recommended_strategy: list[str]  # LLM, domain reasoning over the facts

    # --- audit ---
    qc_flags: list[QCFlag]
    provenance: list[Provenance]
    generated_at: datetime

    model_config = {"json_schema_extra": {"x-llm-writable-fields": ["reasoning", "recommended_strategy", "notes"]}}
