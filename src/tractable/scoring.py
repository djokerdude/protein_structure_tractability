"""Transparent tractability scoring.

The score is an explicit additive rubric, not a black box and not an LLM guess.
Every sub-score is a named, documented function of quantities computed in
``compute``. Weights are v3 defaults declared as module constants and meant to
be *calibrated* against a labelled benchmark of known-tractable vs.
known-intractable targets (see README roadmap) — they are not claimed to be
final.

    total = clamp(
        COVERAGE_WEIGHT      * coverage_fraction
      + DOMAIN_WEIGHT        * solvable_domain_fraction
      + CONFIDENCE_WEIGHT    * (confidence_score / 100)
      + PURIFICATION_WEIGHT  * (purification_score / 100)
      - DISORDER_WEIGHT      * disordered_fraction,
        0, 100,
    )

Interpretation of the five terms:

* coverage          how much of the chain already has experimental structure.
* solvable domains  fraction of annotated folded domains that are either solved
                    or compact + high-confidence, i.e. separately expressible.
* confidence        when experimental structures exist: fraction of those
                    structures with resolution < HIGH_RES_THRESHOLD_A (×100).
                    High-resolution structures indicate the protein is amenable
                    to precise structure determination. When no structures exist:
                    falls back to mean AlphaFold pLDDT over uncovered ordered
                    regions, which estimates whether the gaps are likely orderly
                    and fillable.
* purification      deterministic score from LLM-extracted purification protocol
                    data (expression system, chromatography steps, yield, co-
                    expression requirements). Derived from primary citations of
                    high-resolution structures. None when no protocols available.
* disorder penalty  large flexible/disordered fractions hurt full-length
                    tractability (linkers that resist crystallisation and blur
                    in cryo-EM maps).
"""

from __future__ import annotations

from .schema import ScoreBreakdown

RUBRIC_VERSION = "v3-uncalibrated"

COVERAGE_WEIGHT = 35.0
DOMAIN_WEIGHT = 25.0
CONFIDENCE_WEIGHT = 20.0
PURIFICATION_WEIGHT = 20.0
DISORDER_WEIGHT = 20.0  # subtracted

# Structures at or below this resolution (Å) count as high-resolution.
HIGH_RES_THRESHOLD_A = 3.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score(
    *,
    coverage_fraction: float,
    solvable_domain_fraction: float,
    confidence_score: float | None,
    purification_score: float | None,
    disordered_fraction: float,
) -> ScoreBreakdown:
    """Compute the additive tractability score and its breakdown.

    Args:
        coverage_fraction: 0..1, residues with experimental coverage / length.
        solvable_domain_fraction: 0..1, fraction of annotated domains that are
            solved or compact + high-confidence (computed upstream).
        confidence_score: 0..100 unified confidence metric. When experimental
            structures with resolution data exist, this is the fraction of those
            structures below HIGH_RES_THRESHOLD_A multiplied by 100. When no
            such structures exist, falls back to mean AlphaFold pLDDT over
            uncovered ordered regions. None if neither source is available.
        purification_score: 0..100 purification tractability score derived from
            LLM-extracted protocol data (expression system, steps, yield, co-
            expression requirements). None when no protocol data is available
            (treated as 0 points, not as "difficult").
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

    cs = 0.0 if confidence_score is None else _clamp(confidence_score, 0.0, 100.0)
    confidence_points = CONFIDENCE_WEIGHT * (cs / 100.0)

    ps = 0.0 if purification_score is None else _clamp(purification_score, 0.0, 100.0)
    purification_points = PURIFICATION_WEIGHT * (ps / 100.0)

    disorder_penalty = -DISORDER_WEIGHT * disordered_fraction

    total = _clamp(
        coverage_points + domain_points + confidence_points + purification_points + disorder_penalty,
        0.0,
        100.0,
    )

    return ScoreBreakdown(
        coverage_points=round(coverage_points, 2),
        domain_points=round(domain_points, 2),
        confidence_points=round(confidence_points, 2),
        purification_points=round(purification_points, 2),
        disorder_penalty=round(disorder_penalty, 2),
        total=round(total, 2),
        rubric_version=RUBRIC_VERSION,
    )
