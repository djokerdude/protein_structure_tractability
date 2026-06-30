"""Data-source tools exposed to the agent.

Each function is a single, well-typed unit of data acquisition. They are the
*tools* the LLM agent is allowed to call; the agent decides which to invoke and
in what order, but the functions themselves are deterministic, cached, and
provenance-stamped. Keeping them small and pure is what makes the agent
auditable and the pipeline reproducible.

Real endpoints (documented here, implemented incrementally — see README status):

* UniProt REST .............. https://rest.uniprot.org/uniprotkb/search
* RCSB PDB Search/Data API ... https://search.rcsb.org / https://data.rcsb.org
* PDBe SIFTS (residue map) ... https://www.ebi.ac.uk/pdbe/api/mappings/
* AlphaFold DB .............. https://alphafold.ebi.ac.uk/api/prediction/
* NCBI E-utilities .......... https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Every implementation must (1) cache responses on disk keyed by request so tests
run offline and we respect each service's rate limits and usage policy, and
(2) return a Provenance record alongside the data.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import (
    Candidate,
    Domain,
    ExperimentalStructure,
    Provenance,
)

Range = tuple[int, int]


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


def search_uniprot(query: str, limit: int = 10) -> list[Candidate]:
    """Resolve a free-text protein name to candidate UniProt accessions.

    This is the entity-resolution step. The agent reasons over the returned
    candidates (organism, reviewed vs. unreviewed, isoform) to pick one, or
    asks the user to disambiguate when it is genuinely ambiguous.
    """
    raise NotImplementedError


def get_uniprot_entry(accession: str) -> UniProtEntry:
    """Fetch sequence, length, domain features, and disordered regions.

    Domain ranges come from UniProt feature annotations (DOMAIN/REGION);
    disordered regions from UniProt 'Region: Disordered' features, optionally
    augmented by a low-pLDDT proxy from AlphaFold.
    """
    raise NotImplementedError


def get_pdb_structures(accession: str) -> list[ExperimentalStructure]:
    """Find experimental PDB structures for a UniProt accession.

    Uses RCSB search by UniProt accession; residue-level placement onto the
    sequence is resolved via SIFTS (see ``get_sifts_coverage``), not by
    guessing from titles.
    """
    raise NotImplementedError


def get_sifts_coverage(accession: str) -> list[Range]:
    """Per-residue UniProt->PDB coverage ranges from PDBe SIFTS.

    This is the authoritative source for *which residues* are experimentally
    covered, and therefore for the coverage percentage. Do not infer coverage
    from PDB entry metadata.
    """
    raise NotImplementedError


def get_alphafold_plddt(accession: str) -> list[float]:
    """Per-residue AlphaFold pLDDT (index i -> confidence for residue i+1)."""
    raise NotImplementedError


def get_ncbi_record(accession_or_name: str) -> dict:
    """Fetch an NCBI protein record for cross-source QC (e.g. length check).

    Used by the QC layer to flag inconsistencies between databases rather than
    as a primary data source.
    """
    raise NotImplementedError
