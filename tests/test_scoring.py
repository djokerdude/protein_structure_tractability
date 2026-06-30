"""Tests for the transparent scoring rubric (v3)."""

from tractable import scoring


def test_perfect_score_is_capped():
    s = scoring.score(
        coverage_fraction=1.0,
        solvable_domain_fraction=1.0,
        confidence_score=100.0,
        purification_score=100.0,
        disordered_fraction=0.0,
    )
    assert s.total == 100.0


def test_floor_at_zero():
    s = scoring.score(
        coverage_fraction=0.0,
        solvable_domain_fraction=0.0,
        confidence_score=0.0,
        purification_score=0.0,
        disordered_fraction=1.0,
    )
    assert s.total == 0.0
    assert s.disorder_penalty == -20.0


def test_breakdown_sums_to_total_when_unclamped():
    s = scoring.score(
        coverage_fraction=0.42,
        solvable_domain_fraction=0.67,
        confidence_score=75.0,
        purification_score=60.0,
        disordered_fraction=0.15,
    )
    expected = (
        s.coverage_points
        + s.domain_points
        + s.confidence_points
        + s.purification_points
        + s.disorder_penalty
    )
    assert abs(s.total - round(expected, 2)) < 0.011


def test_none_confidence_score_contributes_zero():
    s = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        confidence_score=None,
        purification_score=None,
        disordered_fraction=0.0,
    )
    assert s.confidence_points == 0.0


def test_none_purification_score_contributes_zero():
    with_purification = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        confidence_score=80.0,
        purification_score=80.0,
        disordered_fraction=0.0,
    )
    without_purification = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        confidence_score=80.0,
        purification_score=None,
        disordered_fraction=0.0,
    )
    assert without_purification.purification_points == 0.0
    assert with_purification.purification_points > 0.0
    assert with_purification.total > without_purification.total


def test_rejects_out_of_range_fraction():
    try:
        scoring.score(
            coverage_fraction=1.5,
            solvable_domain_fraction=0.5,
            confidence_score=80.0,
            purification_score=80.0,
            disordered_fraction=0.0,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for coverage_fraction > 1")


def test_rubric_version_propagates():
    s = scoring.score(
        coverage_fraction=0.5,
        solvable_domain_fraction=0.5,
        confidence_score=50.0,
        purification_score=50.0,
        disordered_fraction=0.1,
    )
    assert s.rubric_version == scoring.RUBRIC_VERSION
