"""Tests for the deterministic geometry. No network, no LLM, no fixtures."""

from tractable import compute


def test_merge_overlapping_and_adjacent():
    assert compute.merge_ranges([(1, 10), (5, 20), (25, 30), (31, 40)]) == [(1, 20), (25, 40)]


def test_merge_empty():
    assert compute.merge_ranges([]) == []


def test_merge_single():
    assert compute.merge_ranges([(3, 7)]) == [(3, 7)]


def test_covered_residue_count_dedups_overlap():
    assert compute.covered_residue_count([(1, 10), (5, 20)]) == 20


def test_coverage_fraction_caps_at_one():
    # overlapping ranges summing past the length must not exceed 1.0
    assert compute.coverage_fraction([(1, 100), (50, 150)], 100) == 1.0


def test_coverage_fraction_basic():
    assert compute.coverage_fraction([(1, 42)], 100) == 0.42


def test_coverage_fraction_rejects_bad_length():
    try:
        compute.coverage_fraction([(1, 10)], 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive length")


def test_domain_coverage_fraction_partial():
    # domain 100..200 (101 residues), covered 100..150 (51 residues)
    frac = compute.domain_coverage_fraction((100, 200), [(100, 150)])
    assert abs(frac - 51 / 101) < 1e-9


def test_is_solved_threshold():
    assert compute.is_solved((1, 100), [(1, 80)], threshold=0.80) is True
    assert compute.is_solved((1, 100), [(1, 79)], threshold=0.80) is False


def test_subtract_ranges_two_gaps():
    assert compute.subtract_ranges((1, 100), [(1, 40), (60, 80)]) == [(41, 59), (81, 100)]


def test_subtract_ranges_fully_covered():
    assert compute.subtract_ranges((10, 20), [(1, 50)]) == []


def test_subtract_ranges_no_coverage():
    assert compute.subtract_ranges((10, 20), []) == [(10, 20)]


def test_missing_regions_filters_tiny_gaps():
    # a 1-residue gap should be dropped at min_length=5
    annotated = [(1, 100)]
    covered = [(1, 50), (52, 100)]  # gap at 51 only
    assert compute.missing_regions(annotated, covered, min_length=5) == []


def test_missing_regions_keeps_real_gap():
    annotated = [(1, 100)]
    covered = [(1, 40)]
    assert compute.missing_regions(annotated, covered, min_length=5) == [(41, 100)]


def test_disordered_fraction():
    assert compute.disordered_fraction([(1, 15)], 100) == 0.15
