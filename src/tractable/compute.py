"""Deterministic geometry over residue ranges.

Pure functions only: no network, no LLM, no I/O. This is where the numbers in
the report actually come from. Everything operates on 1-indexed, inclusive
``(start, end)`` tuples so it can be unit-tested without pydantic or any
external service.

The LLM is *not* allowed to produce any of these quantities; it consumes them.
"""

from __future__ import annotations

Range = tuple[int, int]  # 1-indexed, inclusive

# A domain counts as "solved" once this fraction of its residues are covered
# by at least one experimental structure. Tunable; surfaced in the report.
SOLVED_THRESHOLD = 0.80


def merge_ranges(ranges: list[Range]) -> list[Range]:
    """Merge overlapping or adjacent inclusive ranges into a minimal set.

    >>> merge_ranges([(1, 10), (5, 20), (25, 30), (31, 40)])
    [(1, 20), (25, 40)]
    >>> merge_ranges([])
    []
    """
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[Range] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        # adjacent (end + 1 == start) counts as contiguous coverage
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def covered_residue_count(covered: list[Range]) -> int:
    """Total residues covered, de-duplicating overlaps.

    >>> covered_residue_count([(1, 10), (5, 20)])
    20
    """
    return sum(end - start + 1 for start, end in merge_ranges(covered))


def coverage_fraction(covered: list[Range], sequence_length: int) -> float:
    """Fraction of the full sequence with at least one experimental structure."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    frac = covered_residue_count(covered) / sequence_length
    return min(frac, 1.0)


def overlap_length(a: Range, covered: list[Range]) -> int:
    """Residues of ``a`` that intersect the merged ``covered`` set."""
    a_start, a_end = a
    total = 0
    for start, end in merge_ranges(covered):
        lo, hi = max(a_start, start), min(a_end, end)
        if lo <= hi:
            total += hi - lo + 1
    return total


def domain_coverage_fraction(domain: Range, covered: list[Range]) -> float:
    """How much of a single domain is experimentally covered (0..1)."""
    d_start, d_end = domain
    d_len = d_end - d_start + 1
    if d_len <= 0:
        raise ValueError("domain range must be non-empty")
    return overlap_length(domain, covered) / d_len


def is_solved(domain: Range, covered: list[Range], threshold: float = SOLVED_THRESHOLD) -> bool:
    """Whether a domain clears the 'solved' coverage threshold."""
    return domain_coverage_fraction(domain, covered) >= threshold


def subtract_ranges(whole: Range, covered: list[Range]) -> list[Range]:
    """Return the sub-spans of ``whole`` NOT covered by ``covered``.

    >>> subtract_ranges((1, 100), [(1, 40), (60, 80)])
    [(41, 59), (81, 100)]
    """
    w_start, w_end = whole
    gaps: list[Range] = []
    cursor = w_start
    for start, end in merge_ranges(covered):
        if end < w_start or start > w_end:
            continue
        start = max(start, w_start)
        end = min(end, w_end)
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= w_end:
        gaps.append((cursor, w_end))
    return gaps


def missing_regions(
    annotated: list[Range],
    covered: list[Range],
    min_length: int = 1,
) -> list[Range]:
    """Annotated regions (domains/disordered spans) lacking experimental coverage.

    Returns the uncovered sub-spans of each annotated region, filtered to those
    at least ``min_length`` residues long (to drop trivial 1-2 residue gaps at
    domain boundaries).
    """
    out: list[Range] = []
    for region in annotated:
        for gap in subtract_ranges(region, covered):
            if gap[1] - gap[0] + 1 >= min_length:
                out.append(gap)
    return merge_ranges(out)


def disordered_fraction(disordered: list[Range], sequence_length: int) -> float:
    """Fraction of the sequence annotated (or predicted) as disordered."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return min(covered_residue_count(disordered) / sequence_length, 1.0)
