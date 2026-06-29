"""Cross-source consistency checks for tractability pipeline data.

Pure functions only: no network, no LLM, no I/O. Each check inspects one
or more data sources and returns a (possibly empty) list of QCFlag instances.
``run_all`` is the entry point used by the pipeline; individual functions are
exposed directly for isolated testing.

All thresholds are module-level constants — inspectable and tunable without
touching logic. Functions return lists (never raise) so callers can accumulate
flags from multiple sources in a single pass.
"""

from __future__ import annotations

from .schema import Domain, ExperimentalStructure, QCFlag, QCSeverity, Source

Range = tuple[int, int]  # 1-indexed, inclusive — mirrors compute.py

# --- Tunable thresholds -------------------------------------------------------

# UniProt vs NCBI: any non-zero diff triggers WARNING; fraction at or above
# this threshold escalates to ERROR.
LENGTH_ERROR_FRACTION: float = 0.05

# Disordered-fraction gates.
HIGH_DISORDER_WARN: float = 0.50   # domain-focused strategy recommended
HIGH_DISORDER_ERROR: float = 0.80  # full-length structural work very unlikely

# Experimental coverage below this fraction warrants a WARNING.
LOW_COVERAGE_WARN: float = 0.25

PLDDT_MIN: float = 0.0
PLDDT_MAX: float = 100.0


# --- Individual checks --------------------------------------------------------


def check_length_consistency(
    uniprot_length: int,
    ncbi_length: int,
) -> list[QCFlag]:
    """Flag a discrepancy between UniProt and NCBI canonical sequence lengths.

    Any non-zero difference triggers at least a WARNING; a difference of
    ≥ LENGTH_ERROR_FRACTION of the UniProt length escalates to ERROR.
    """
    diff = abs(uniprot_length - ncbi_length)
    if diff == 0:
        return []
    fraction = diff / uniprot_length if uniprot_length > 0 else 1.0
    severity = QCSeverity.ERROR if fraction >= LENGTH_ERROR_FRACTION else QCSeverity.WARNING
    return [
        QCFlag(
            severity=severity,
            code="LENGTH_MISMATCH",
            message=(
                f"UniProt length {uniprot_length} differs from NCBI length "
                f"{ncbi_length} by {diff} residues ({fraction:.1%})."
            ),
            sources_involved=[Source.UNIPROT, Source.NCBI],
        )
    ]


def check_sifts_ranges(
    covered_ranges: list[Range],
    sequence_length: int,
) -> list[QCFlag]:
    """Flag SIFTS coverage ranges that fall outside the UniProt sequence boundaries.

    Any range with start < 1, end > sequence_length, or start > end is an ERROR
    because it indicates a residue-numbering mismatch between SIFTS and UniProt.
    """
    flags: list[QCFlag] = []
    for start, end in covered_ranges:
        if start < 1 or end > sequence_length or start > end:
            flags.append(
                QCFlag(
                    severity=QCSeverity.ERROR,
                    code="SIFTS_RANGE_OUT_OF_BOUNDS",
                    message=(
                        f"SIFTS coverage range ({start}, {end}) is outside the "
                        f"valid UniProt range [1, {sequence_length}] or has start > end."
                    ),
                    sources_involved=[Source.PDBE_SIFTS, Source.UNIPROT],
                )
            )
    return flags


def check_domain_ranges(
    domains: list[Domain],
    sequence_length: int,
) -> list[QCFlag]:
    """Flag annotated domains whose ranges are invalid or exceed the sequence length."""
    flags: list[QCFlag] = []
    for domain in domains:
        start, end = domain.range.start, domain.range.end
        if start > end or start < 1 or end > sequence_length:
            flags.append(
                QCFlag(
                    severity=QCSeverity.ERROR,
                    code="DOMAIN_RANGE_INVALID",
                    message=(
                        f"Domain '{domain.name}' range ({start}, {end}) is invalid "
                        f"or outside sequence length {sequence_length}."
                    ),
                    sources_involved=[Source.UNIPROT],
                )
            )
    return flags


def check_disordered_ranges(
    disordered_ranges: list[Range],
    sequence_length: int,
) -> list[QCFlag]:
    """Flag disordered region annotations that fall outside the sequence."""
    flags: list[QCFlag] = []
    for start, end in disordered_ranges:
        if start < 1 or end > sequence_length or start > end:
            flags.append(
                QCFlag(
                    severity=QCSeverity.ERROR,
                    code="DISORDERED_RANGE_OUT_OF_BOUNDS",
                    message=(
                        f"Disordered region ({start}, {end}) is outside the "
                        f"valid sequence range [1, {sequence_length}] or has start > end."
                    ),
                    sources_involved=[Source.UNIPROT],
                )
            )
    return flags


def check_no_structures(
    structures: list[ExperimentalStructure],
) -> list[QCFlag]:
    """Warn if no experimental PDB structures were found for the accession."""
    if structures:
        return []
    return [
        QCFlag(
            severity=QCSeverity.WARNING,
            code="NO_STRUCTURES_FOUND",
            message="No experimental PDB structures found for this accession.",
            sources_involved=[Source.RCSB_PDB, Source.PDBE_SIFTS],
        )
    ]


def check_plddt_vector(
    plddt: list[float],
    sequence_length: int,
) -> list[QCFlag]:
    """Flag AlphaFold pLDDT vectors with the wrong length or out-of-range values.

    Both checks run independently so both errors are reported in one pass.
    """
    flags: list[QCFlag] = []
    if len(plddt) != sequence_length:
        flags.append(
            QCFlag(
                severity=QCSeverity.ERROR,
                code="PLDDT_LENGTH_MISMATCH",
                message=(
                    f"AlphaFold pLDDT vector has {len(plddt)} values but the "
                    f"UniProt sequence length is {sequence_length}."
                ),
                sources_involved=[Source.ALPHAFOLD, Source.UNIPROT],
            )
        )
    out_of_range = [v for v in plddt if not (PLDDT_MIN <= v <= PLDDT_MAX)]
    if out_of_range:
        flags.append(
            QCFlag(
                severity=QCSeverity.ERROR,
                code="PLDDT_VALUES_OUT_OF_RANGE",
                message=(
                    f"{len(out_of_range)} pLDDT value(s) outside [{PLDDT_MIN}, "
                    f"{PLDDT_MAX}] (first bad value: {out_of_range[0]:.2f})."
                ),
                sources_involved=[Source.ALPHAFOLD],
            )
        )
    return flags


def check_high_disorder(disordered_fraction: float) -> list[QCFlag]:
    """Warn or error on proteins with large disordered fractions.

    Escalates to ERROR at HIGH_DISORDER_ERROR because high disorder makes
    full-length crystallisation and cryo-EM extremely difficult.
    """
    if disordered_fraction >= HIGH_DISORDER_ERROR:
        return [
            QCFlag(
                severity=QCSeverity.ERROR,
                code="HIGH_DISORDER",
                message=(
                    f"Disordered fraction {disordered_fraction:.1%} exceeds the "
                    f"error threshold ({HIGH_DISORDER_ERROR:.0%}); full-length "
                    "structural work is very unlikely to succeed."
                ),
                sources_involved=[Source.UNIPROT],
            )
        ]
    if disordered_fraction >= HIGH_DISORDER_WARN:
        return [
            QCFlag(
                severity=QCSeverity.WARNING,
                code="HIGH_DISORDER",
                message=(
                    f"Disordered fraction {disordered_fraction:.1%} exceeds "
                    f"{HIGH_DISORDER_WARN:.0%}; consider a domain-focused strategy."
                ),
                sources_involved=[Source.UNIPROT],
            )
        ]
    return []


def check_low_coverage(coverage_fraction: float) -> list[QCFlag]:
    """Warn if experimental residue coverage is below LOW_COVERAGE_WARN."""
    if coverage_fraction < LOW_COVERAGE_WARN:
        return [
            QCFlag(
                severity=QCSeverity.WARNING,
                code="LOW_COVERAGE",
                message=(
                    f"Experimental coverage {coverage_fraction:.1%} is below "
                    f"{LOW_COVERAGE_WARN:.0%}; large portions of the chain are unexplored."
                ),
                sources_involved=[Source.PDBE_SIFTS],
            )
        ]
    return []


# --- Aggregate entry point ----------------------------------------------------


def run_all(
    *,
    uniprot_length: int,
    ncbi_length: int | None = None,
    covered_ranges: list[Range],
    disordered_ranges: list[Range],
    structures: list[ExperimentalStructure],
    domains: list[Domain],
    plddt: list[float] | None = None,
    coverage_fraction: float,
    disordered_fraction: float,
) -> list[QCFlag]:
    """Run all configured QC checks and return every flag produced.

    Optional arguments (``ncbi_length``, ``plddt``) are silently skipped when
    None — their source was not fetched in this run.
    """
    flags: list[QCFlag] = []

    if ncbi_length is not None:
        flags.extend(check_length_consistency(uniprot_length, ncbi_length))

    flags.extend(check_sifts_ranges(covered_ranges, uniprot_length))
    flags.extend(check_domain_ranges(domains, uniprot_length))
    flags.extend(check_disordered_ranges(disordered_ranges, uniprot_length))
    flags.extend(check_no_structures(structures))

    if plddt is not None:
        flags.extend(check_plddt_vector(plddt, uniprot_length))

    flags.extend(check_high_disorder(disordered_fraction))
    flags.extend(check_low_coverage(coverage_fraction))

    return flags
