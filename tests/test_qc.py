"""Tests for qc.py — all offline, no network, no LLM, no fixtures."""

from datetime import datetime

import pytest

from tractable import qc
from tractable.schema import (
    Domain,
    ExperimentalStructure,
    Provenance,
    QCSeverity,
    ResidueRange,
    Source,
)


# --- Helpers ------------------------------------------------------------------


def _prov(source: Source = Source.RCSB_PDB) -> Provenance:
    return Provenance(source=source, identifier="test", retrieved_at=datetime(2024, 1, 1))


def _structure(start: int, end: int, pdb_id: str = "1ABC") -> ExperimentalStructure:
    return ExperimentalStructure(
        pdb_id=pdb_id,
        method="X-RAY DIFFRACTION",
        resolution_a=2.0,
        covered_range=ResidueRange(start=start, end=end),
        provenance=_prov(),
    )


def _domain(name: str, start: int, end: int) -> Domain:
    return Domain(
        name=name,
        range=ResidueRange(start=start, end=end),
        source=Source.UNIPROT,
        coverage_fraction=0.0,
        solved=False,
    )


# --- check_length_consistency -------------------------------------------------


def test_length_consistency_identical():
    assert qc.check_length_consistency(500, 500) == []


def test_length_consistency_warning_small_diff():
    flags = qc.check_length_consistency(1000, 999)
    assert len(flags) == 1
    assert flags[0].code == "LENGTH_MISMATCH"
    assert flags[0].severity == QCSeverity.WARNING
    assert Source.UNIPROT in flags[0].sources_involved
    assert Source.NCBI in flags[0].sources_involved


def test_length_consistency_error_large_diff():
    # 500 vs 450 is 10% → ERROR
    flags = qc.check_length_consistency(500, 450)
    assert len(flags) == 1
    assert flags[0].severity == QCSeverity.ERROR


def test_length_consistency_error_at_fraction_boundary():
    # exactly 5% of 200 = 10 residues → ERROR (≥ threshold)
    flags = qc.check_length_consistency(200, 190)
    assert flags[0].severity == QCSeverity.ERROR


def test_length_consistency_warning_just_below_fraction():
    # 4 residues off on a 200-residue protein → 2% → WARNING
    flags = qc.check_length_consistency(200, 196)
    assert flags[0].severity == QCSeverity.WARNING


def test_length_consistency_message_contains_diff():
    flags = qc.check_length_consistency(500, 450)
    assert "50" in flags[0].message


# --- check_sifts_ranges -------------------------------------------------------


def test_sifts_ranges_all_valid():
    assert qc.check_sifts_ranges([(1, 100), (200, 300)], 500) == []


def test_sifts_ranges_start_zero():
    flags = qc.check_sifts_ranges([(0, 50)], 500)
    assert len(flags) == 1
    assert flags[0].code == "SIFTS_RANGE_OUT_OF_BOUNDS"
    assert flags[0].severity == QCSeverity.ERROR


def test_sifts_ranges_end_exceeds_length():
    flags = qc.check_sifts_ranges([(400, 501)], 500)
    assert flags[0].code == "SIFTS_RANGE_OUT_OF_BOUNDS"


def test_sifts_ranges_start_greater_than_end():
    flags = qc.check_sifts_ranges([(100, 50)], 500)
    assert flags[0].code == "SIFTS_RANGE_OUT_OF_BOUNDS"


def test_sifts_ranges_empty_list():
    assert qc.check_sifts_ranges([], 500) == []


def test_sifts_ranges_multiple_bad_ranges():
    flags = qc.check_sifts_ranges([(0, 10), (1, 50), (490, 600)], 500)
    # two bad, one good
    assert len(flags) == 2


def test_sifts_ranges_exact_boundaries_valid():
    assert qc.check_sifts_ranges([(1, 500)], 500) == []


# --- check_domain_ranges ------------------------------------------------------


def test_domain_ranges_valid():
    domains = [_domain("Kinase", 10, 300), _domain("SH2", 310, 400)]
    assert qc.check_domain_ranges(domains, 500) == []


def test_domain_ranges_exceeds_length():
    flags = qc.check_domain_ranges([_domain("Big", 1, 600)], 500)
    assert len(flags) == 1
    assert flags[0].code == "DOMAIN_RANGE_INVALID"
    assert flags[0].severity == QCSeverity.ERROR


def test_domain_ranges_empty():
    assert qc.check_domain_ranges([], 500) == []


def test_domain_ranges_message_contains_name():
    flags = qc.check_domain_ranges([_domain("MyDomain", 1, 600)], 500)
    assert "MyDomain" in flags[0].message


# --- check_disordered_ranges --------------------------------------------------


def test_disordered_ranges_valid():
    assert qc.check_disordered_ranges([(1, 50), (200, 300)], 500) == []


def test_disordered_ranges_start_zero():
    flags = qc.check_disordered_ranges([(0, 20)], 500)
    assert len(flags) == 1
    assert flags[0].code == "DISORDERED_RANGE_OUT_OF_BOUNDS"


def test_disordered_ranges_end_exceeds_length():
    flags = qc.check_disordered_ranges([(450, 510)], 500)
    assert flags[0].code == "DISORDERED_RANGE_OUT_OF_BOUNDS"


def test_disordered_ranges_start_gt_end():
    flags = qc.check_disordered_ranges([(80, 20)], 500)
    assert flags[0].code == "DISORDERED_RANGE_OUT_OF_BOUNDS"


def test_disordered_ranges_empty():
    assert qc.check_disordered_ranges([], 500) == []


# --- check_no_structures ------------------------------------------------------


def test_no_structures_empty_list():
    flags = qc.check_no_structures([])
    assert len(flags) == 1
    assert flags[0].code == "NO_STRUCTURES_FOUND"
    assert flags[0].severity == QCSeverity.WARNING
    assert Source.RCSB_PDB in flags[0].sources_involved


def test_no_structures_with_one_structure():
    assert qc.check_no_structures([_structure(1, 200)]) == []


def test_no_structures_with_multiple_structures():
    assert qc.check_no_structures([_structure(1, 100), _structure(150, 300)]) == []


# --- check_plddt_vector -------------------------------------------------------


def test_plddt_vector_valid():
    assert qc.check_plddt_vector([85.0] * 100, 100) == []


def test_plddt_vector_boundary_values():
    # 0.0 and 100.0 are valid boundary values
    assert qc.check_plddt_vector([0.0, 100.0] + [50.0] * 98, 100) == []


def test_plddt_vector_length_mismatch():
    flags = qc.check_plddt_vector([85.0] * 90, 100)
    codes = {f.code for f in flags}
    assert "PLDDT_LENGTH_MISMATCH" in codes
    assert flags[0].severity == QCSeverity.ERROR


def test_plddt_vector_values_above_max():
    plddt = [85.0] * 99 + [105.0]
    flags = qc.check_plddt_vector(plddt, 100)
    codes = {f.code for f in flags}
    assert "PLDDT_VALUES_OUT_OF_RANGE" in codes


def test_plddt_vector_values_below_min():
    plddt = [-1.0] + [85.0] * 99
    flags = qc.check_plddt_vector(plddt, 100)
    codes = {f.code for f in flags}
    assert "PLDDT_VALUES_OUT_OF_RANGE" in codes


def test_plddt_vector_both_errors_reported():
    # wrong length AND bad values — both flags should appear
    plddt = [150.0] * 50
    flags = qc.check_plddt_vector(plddt, 100)
    codes = {f.code for f in flags}
    assert "PLDDT_LENGTH_MISMATCH" in codes
    assert "PLDDT_VALUES_OUT_OF_RANGE" in codes


def test_plddt_vector_empty():
    flags = qc.check_plddt_vector([], 100)
    codes = {f.code for f in flags}
    assert "PLDDT_LENGTH_MISMATCH" in codes


# --- check_high_disorder ------------------------------------------------------


def test_high_disorder_low_fraction():
    assert qc.check_high_disorder(0.10) == []


def test_high_disorder_below_warn_threshold():
    assert qc.check_high_disorder(qc.HIGH_DISORDER_WARN - 0.01) == []


def test_high_disorder_at_warn_threshold():
    flags = qc.check_high_disorder(qc.HIGH_DISORDER_WARN)
    assert len(flags) == 1
    assert flags[0].code == "HIGH_DISORDER"
    assert flags[0].severity == QCSeverity.WARNING


def test_high_disorder_between_thresholds():
    flags = qc.check_high_disorder(0.65)
    assert flags[0].severity == QCSeverity.WARNING


def test_high_disorder_at_error_threshold():
    flags = qc.check_high_disorder(qc.HIGH_DISORDER_ERROR)
    assert flags[0].severity == QCSeverity.ERROR


def test_high_disorder_above_error_threshold():
    flags = qc.check_high_disorder(0.95)
    assert flags[0].severity == QCSeverity.ERROR


def test_high_disorder_sources_uniprot():
    flags = qc.check_high_disorder(0.90)
    assert Source.UNIPROT in flags[0].sources_involved


# --- check_low_coverage -------------------------------------------------------


def test_low_coverage_high_fraction():
    assert qc.check_low_coverage(0.80) == []


def test_low_coverage_at_threshold_no_flag():
    # exactly at threshold is OK (check is strict <)
    assert qc.check_low_coverage(qc.LOW_COVERAGE_WARN) == []


def test_low_coverage_just_below_threshold():
    flags = qc.check_low_coverage(qc.LOW_COVERAGE_WARN - 0.01)
    assert len(flags) == 1
    assert flags[0].code == "LOW_COVERAGE"
    assert flags[0].severity == QCSeverity.WARNING


def test_low_coverage_zero():
    flags = qc.check_low_coverage(0.0)
    assert flags[0].code == "LOW_COVERAGE"


def test_low_coverage_sources_sifts():
    flags = qc.check_low_coverage(0.10)
    assert Source.PDBE_SIFTS in flags[0].sources_involved


# --- run_all ------------------------------------------------------------------


def test_run_all_clean_data_produces_no_flags():
    """Fully consistent data should produce no QC flags."""
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=300,
        covered_ranges=[(1, 200)],
        disordered_ranges=[(220, 240)],
        structures=[_structure(1, 200)],
        domains=[_domain("Kinase", 10, 200)],
        plddt=[75.0] * 300,
        coverage_fraction=200 / 300,
        disordered_fraction=21 / 300,
    )
    assert flags == []


def test_run_all_ncbi_none_skips_length_check():
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=None,
        covered_ranges=[(1, 200)],
        disordered_ranges=[],
        structures=[_structure(1, 200)],
        domains=[],
        plddt=None,
        coverage_fraction=200 / 300,
        disordered_fraction=0.0,
    )
    codes = {f.code for f in flags}
    assert "LENGTH_MISMATCH" not in codes


def test_run_all_plddt_none_skips_plddt_checks():
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=300,
        covered_ranges=[(1, 200)],
        disordered_ranges=[],
        structures=[_structure(1, 200)],
        domains=[],
        plddt=None,
        coverage_fraction=200 / 300,
        disordered_fraction=0.0,
    )
    codes = {f.code for f in flags}
    assert not any(c.startswith("PLDDT") for c in codes)


def test_run_all_accumulates_multiple_flags():
    """Multiple simultaneous problems should each produce their own flag."""
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=150,     # large mismatch → ERROR
        covered_ranges=[],
        disordered_ranges=[],
        structures=[],       # no structures → WARNING
        domains=[],
        plddt=None,
        coverage_fraction=0.0,  # low coverage → WARNING
        disordered_fraction=0.85,  # high disorder → ERROR
    )
    codes = {f.code for f in flags}
    assert "LENGTH_MISMATCH" in codes
    assert "NO_STRUCTURES_FOUND" in codes
    assert "LOW_COVERAGE" in codes
    assert "HIGH_DISORDER" in codes


def test_run_all_sifts_bad_range_propagates():
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=300,
        covered_ranges=[(0, 100)],  # start=0 is invalid
        disordered_ranges=[],
        structures=[_structure(1, 100)],
        domains=[],
        plddt=None,
        coverage_fraction=100 / 300,
        disordered_fraction=0.0,
    )
    codes = {f.code for f in flags}
    assert "SIFTS_RANGE_OUT_OF_BOUNDS" in codes


def test_run_all_domain_bad_range_propagates():
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=300,
        covered_ranges=[(1, 200)],
        disordered_ranges=[],
        structures=[_structure(1, 200)],
        domains=[_domain("Overflow", 1, 500)],
        plddt=None,
        coverage_fraction=200 / 300,
        disordered_fraction=0.0,
    )
    codes = {f.code for f in flags}
    assert "DOMAIN_RANGE_INVALID" in codes


def test_run_all_severities_reflect_individual_checks():
    """run_all must not modify severity relative to individual check functions."""
    flags = qc.run_all(
        uniprot_length=300,
        ncbi_length=150,  # 50% off → ERROR
        covered_ranges=[(1, 200)],
        disordered_ranges=[],
        structures=[_structure(1, 200)],
        domains=[],
        plddt=None,
        coverage_fraction=200 / 300,
        disordered_fraction=0.0,
    )
    mismatch = next(f for f in flags if f.code == "LENGTH_MISMATCH")
    assert mismatch.severity == QCSeverity.ERROR
