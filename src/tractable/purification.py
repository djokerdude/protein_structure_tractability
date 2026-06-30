"""Deterministic purification tractability scoring.

Pure functions only: no network, no LLM. Scores a list of
PurificationProtocol objects extracted by the agent and returns a 0..100
aggregate representing how easily the protein can be produced in quantities
and purity suitable for structural biology.

Score components per protocol:
  - Expression system base score (bacterial > yeast > insect > mammalian)
  - Step penalty: each chromatography step beyond 2 costs 5 points
  - Co-expression penalty: -15 when a partner is required for stability
  - Yield adjustment: modifier when yield is mentioned in the paper

The aggregate is the mean across all extracted protocols.  When no protocols
are available the function returns None (caller decides how to handle the
missing-data case in scoring).
"""

from __future__ import annotations

from .schema import ExpressionSystem, PurificationProtocol

# Base tractability score (0–100) per expression system.
# Reflects practical difficulty: bacterial expression is fast, cheap, and
# yields are typically high; mammalian expression requires specialised
# equipment, longer timelines, and is substantially more expensive.
_EXPRESSION_BASE: dict[ExpressionSystem, float] = {
    ExpressionSystem.ECOLI: 100.0,
    ExpressionSystem.YEAST: 72.0,
    ExpressionSystem.CELL_FREE: 65.0,
    ExpressionSystem.INSECT: 50.0,
    ExpressionSystem.MAMMALIAN: 25.0,
    ExpressionSystem.UNKNOWN: 50.0,  # neutral when system not identified
}

# Points added or subtracted for stated yield.
_YIELD_ADJUSTMENT: dict[str, float] = {
    "high": 5.0,
    "medium": 0.0,
    "low": -15.0,
    "unknown": 0.0,
}

# Points deducted for each chromatography step beyond the first two.
_STEP_PENALTY = 5.0

# Points deducted when a co-expression partner is required.
_COEXPRESSION_PENALTY = 15.0


def _score_one(p: PurificationProtocol) -> float:
    """Score a single purification protocol (0..100)."""
    base = _EXPRESSION_BASE.get(p.expression_system, 50.0)
    step_penalty = max(0.0, len(p.purification_steps) - 2) * _STEP_PENALTY
    coexp_penalty = _COEXPRESSION_PENALTY if p.requires_coexpression else 0.0
    yield_adj = _YIELD_ADJUSTMENT.get(p.yield_category, 0.0)
    return max(0.0, min(100.0, base - step_penalty - coexp_penalty + yield_adj))


def purification_tractability_score(
    protocols: list[PurificationProtocol],
) -> float | None:
    """Return 0..100 aggregate purification tractability, or None if no data.

    None is returned (not zero) when the list is empty so the caller can
    distinguish "not studied" from "studied and difficult".
    """
    if not protocols:
        return None
    scores = [_score_one(p) for p in protocols]
    return round(sum(scores) / len(scores), 2)
