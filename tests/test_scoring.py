"""Tests for the transparent scoring rubric."""

from tractable import scoring


def test_perfect_score_is_capped():
    s = scoring.score(
        coverage_fraction=1.0,
        solvable_domain_fraction=1.0,
        mean_plddt_uncovered=100.0,
        disordered_fraction=0.0,
    )
    assert s.total == 100.0


def test_floor_at_zero():
    # heavy disorder, nothing else, must not go negative
    s = scoring.score(
        coverage_fraction=0.0,
        solvable_domain_fraction=0.0,
        mean_plddt_uncovered=0.0,
        disordered_fraction=1.0,
    )
    assert s.total == 0.0
    assert s.disorder_penalty == -20.0


def test_breakdown_sums_to_total_when_unclamped():
    s = scoring.score(
        coverage_fraction=0.42,
        solvable_domain_fraction=0.67,
        mean_plddt_uncovered=75.0,
        disordered_fraction=0.15,
    )
    expected = (
        s.coverage_points + s.domain_points + s.confidence_points + s.disorder_penalty
    )
    assert abs(s.total - round(expected, 2)) < 0.011


def test_none_plddt_contributes_zero_confidence():
    s = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        mean_plddt_uncovered=None,
        disordered_fraction=0.0,
    )
    assert s.confidence_points == 0.0


def test_rejects_out_of_range_fraction():
    try:
        scoring.score(
            coverage_fraction=1.5,
            solvable_domain_fraction=0.5,
            mean_plddt_uncovered=80.0,
            disordered_fraction=0.0,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for coverage_fraction > 1")


def test_rubric_version_propagates():
    s = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        mean_plddt_uncovered=50.0,
        disordered_fraction=0.1,
    )
    assert s.rubric_version == scoring.RUBRIC_VERSION
