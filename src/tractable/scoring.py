"""Transparent tractability scoring.

The score is an explicit additive rubric, not a black box and not an LLM guess.
Every sub-score is a named, documented function of quantities computed in
``compute``. Weights are v1 defaults declared as module constants and meant to
be *calibrated* against a labelled benchmark of known-tractable vs.
known-intractable targets (see README roadmap) — they are not claimed to be
final.

    total = clamp(
        COVERAGE_WEIGHT      * coverage_fraction
      + DOMAIN_WEIGHT        * solvable_domain_fraction
      + CONFIDENCE_WEIGHT    * (mean_plddt_uncovered / 100)
      - DISORDER_WEIGHT      * disordered_fraction,
        0, 100,
    )

Interpretation of the four terms:

* coverage          how much of the chain already has experimental structure.
* solvable domains  fraction of annotated folded domains that are either solved
                    or compact + high-confidence, i.e. separately expressible.
* gap confidence    AlphaFold pLDDT over the *uncovered* ordered regions; high
                    confidence means the gaps are likely orderly and fillable.
* disorder penalty  large flexible/disordered fractions hurt full-length
                    tractability (linkers that resist crystallisation and blur
                    in cryo-EM maps).
"""

from __future__ import annotations

from .schema import ScoreBreakdown

RUBRIC_VERSION = "v1-uncalibrated"

COVERAGE_WEIGHT = 45.0
DOMAIN_WEIGHT = 30.0
CONFIDENCE_WEIGHT = 25.0
DISORDER_WEIGHT = 20.0  # subtracted


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score(
    *,
    coverage_fraction: float,
    solvable_domain_fraction: float,
    mean_plddt_uncovered: float | None,
    disordered_fraction: float,
) -> ScoreBreakdown:
    """Compute the additive tractability score and its breakdown.

    Args:
        coverage_fraction: 0..1, residues with experimental coverage / length.
        solvable_domain_fraction: 0..1, fraction of annotated domains that are
            solved or compact + high-confidence (computed upstream).
        mean_plddt_uncovered: mean AlphaFold pLDDT (0..100) over uncovered
            ordered regions, or None if there are no such regions / no model.
        disordered_fraction: 0..1, fraction of the chain that is disordered.
    """
    for name, val in (
        ("coverage_fraction", coverage_fraction),
        ("solvable_domain_fraction", solvable_domain_fraction),
        ("disordered_fraction", disordered_fraction),
    ):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    coverage_points = COVERAGE_WEIGHT * coverage_fraction
    domain_points = DOMAIN_WEIGHT * solvable_domain_fraction
    # If there are no uncovered ordered regions, there is no gap to score and the
    # confidence term contributes nothing (neither reward nor penalty).
    plddt = 0.0 if mean_plddt_uncovered is None else _clamp(mean_plddt_uncovered, 0.0, 100.0)
    confidence_points = CONFIDENCE_WEIGHT * (plddt / 100.0)
    disorder_penalty = -DISORDER_WEIGHT * disordered_fraction

    total = _clamp(
        coverage_points + domain_points + confidence_points + disorder_penalty,
        0.0,
        100.0,
    )

    return ScoreBreakdown(
        coverage_points=round(coverage_points, 2),
        domain_points=round(domain_points, 2),
        confidence_points=round(confidence_points, 2),
        disorder_penalty=round(disorder_penalty, 2),
        total=round(total, 2),
        rubric_version=RUBRIC_VERSION,
    )
