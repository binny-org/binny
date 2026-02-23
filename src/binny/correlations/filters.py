"""Tuple-level filters for index-tuples.

Filters are generic: they operate on index tuples and accept precomputed
per-position scores or user-supplied metric callables.

A tuple is an ordered sequence of integer indices, e.g. (i, j) for correlations
or (i, j, k) for triples. For tuple-aware filters, a "position" refers to the
slot within the tuple (0-based) rather than any physical meaning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from binny.utils.types import IndexTuple, IndexTuples

Relations = Literal["lt", "le", "gt", "ge"]

__all__ = [
    "filter_by_score_relation",
    "filter_by_metric_threshold",
    "filter_by_score_separation",
    "filter_by_score_difference",
    "filter_by_score_consistency",
    "filter_by_width_ratio",
    "filter_by_curve_norm_threshold",
]


def _apply_comparator(a: float, b: float, op: str) -> bool:
    """Apply a string comparator token to two floats.

    Supported tokens:
        - "lt": a < b
        - "le": a <= b
        - "gt": a > b
        - "ge": a >= b

    Args:
        a: Left-hand side value.
        b: Right-hand side value.
        op: Comparator token ("lt", "le", "gt", "ge").

    Returns:
        Result of the requested comparison.

    Raises:
        ValueError: If op is not a supported comparator token.
    """
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    raise ValueError(f"Unknown comparator: {op!r}")


def _require_pos(scores: Sequence[Mapping[int, float]], pos: int) -> None:
    """Check that a position is valid for a sequence of score mappings."""
    if pos < 0 or pos >= len(scores):
        raise ValueError(f"pos={pos} is out of range for scores with {len(scores)} positions.")


def _require_same_len(a: Sequence[object], b: Sequence[object], name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} must have the same length.")


def filter_by_score_relation(
    tuples: Sequence[IndexTuple],
    *,
    scores: Sequence[Mapping[int, float]],
    pos_a: int = 0,
    pos_b: int = 1,
    relation: Relations = "lt",
) -> IndexTuples:
    """Filter index tuples by a relation between two position scores.

    This filter generalizes pair-wise score comparison to arbitrary tuples by
    comparing the score at one tuple position to the score at another.

    For each tuple `t`, compare the score associated with `t[pos_b]` to the score
    associated with `t[pos_a]`, using the requested relation. The tuple is kept
    when that comparison passes.

    Args:
        tuples: Sequence of index tuples to be filtered.
        scores: Sequence of per-position score mappings. scores[p] maps the
            index value at tuple position p to a scalar score.
        pos_a: First tuple position in the comparison.
        pos_b: Second tuple position in the comparison.
        relation: Comparison operator applied between the scores at `t[pos_b]`
            and `t[pos_a]`.

    Returns:
        List of index tuples that satisfy the requested score relation.

    Raises:
        ValueError: If pos_a or pos_b is out of range, or tuple is too short.
        KeyError: If a required index is missing from the score mappings.
    """
    _require_pos(scores, pos_a)
    _require_pos(scores, pos_b)

    out: IndexTuples = []
    for t in tuples:
        if pos_a >= len(t) or pos_b >= len(t):
            raise ValueError("Tuple length is smaller than requested positions.")
        i = int(t[pos_a])
        j = int(t[pos_b])
        if i not in scores[pos_a]:
            raise KeyError(f"Missing score for position {pos_a}, index {i}.")
        if j not in scores[pos_b]:
            raise KeyError(f"Missing score for position {pos_b}, index {j}.")
        if _apply_comparator(float(scores[pos_b][j]), float(scores[pos_a][i]), relation):
            out.append(tuple(int(x) for x in t))
    return out


def filter_by_metric_threshold(
    tuples: Sequence[IndexTuple],
    *,
    metric: Callable[..., float],
    threshold: float,
    compare: Relations = "le",
) -> IndexTuples:
    """Filter index tuples using an n-ary metric threshold.

    Each tuple is evaluated using the supplied metric function and retained if
    the resulting value satisfies the requested comparison against the given
    threshold.

    The metric is called as metric(*t), where t is the tuple.

    Args:
        tuples: Sequence of index tuples to be filtered.
        metric: Callable returning a scalar metric value for a given tuple.
        threshold: Reference value used for filtering.
        compare: Comparison operator applied as metric(*t) op threshold.

    Returns:
        List of index tuples that satisfy the metric threshold condition.

    Raises:
        ValueError: If the comparison operator is not recognized.
    """
    thr = float(threshold)
    out: IndexTuples = []
    for t in tuples:
        val = float(metric(*t))
        if _apply_comparator(val, thr, compare):
            out.append(tuple(int(x) for x in t))
    return out


def filter_by_score_separation(
    tuples: Sequence[IndexTuple],
    *,
    scores: Sequence[Mapping[int, float]],
    pos_a: int = 0,
    pos_b: int = 1,
    min_sep: float | None = None,
    max_sep: float | None = None,
    absolute: bool = True,
) -> IndexTuples:
    """Filter index tuples by separation between two position scores.

    This filter keeps tuples whose score separation lies within a requested
    window. It is useful when the score encodes a physical location, such as a
    peak, mean, or median along a redshift grid.

    For each tuple `t`, compute the separation between the scores at `t[pos_a]`
    and `t[pos_b]`. If `absolute=True`, the absolute separation is used. The tuple
    is kept when the separation falls within the requested bounds (ignoring bounds
    set to `None`).

    Args:
        tuples: Sequence of index tuples to be filtered.
        scores: Sequence of per-position score mappings.
        pos_a: First tuple position in the separation.
        pos_b: Second tuple position in the separation.
        min_sep: Optional minimum separation to enforce.
        max_sep: Optional maximum separation to enforce.
        absolute: If True, apply bounds to the absolute separation.

    Returns:
        List of index tuples that satisfy the separation bounds.

    Raises:
        ValueError: If positions are invalid, tuple is too short, or bounds are
            negative.
        KeyError: If a required index is missing from the score mappings.
    """
    _require_pos(scores, pos_a)
    _require_pos(scores, pos_b)

    if min_sep is not None and min_sep < 0.0:
        raise ValueError("min_sep must be >= 0 when provided.")
    if max_sep is not None and max_sep < 0.0:
        raise ValueError("max_sep must be >= 0 when provided.")

    out: IndexTuples = []
    for t in tuples:
        if pos_a >= len(t) or pos_b >= len(t):
            raise ValueError("Tuple length is smaller than requested positions.")
        i = int(t[pos_a])
        j = int(t[pos_b])

        if i not in scores[pos_a]:
            raise KeyError(f"Missing score for position {pos_a}, index {i}.")
        if j not in scores[pos_b]:
            raise KeyError(f"Missing score for position {pos_b}, index {j}.")

        d = float(scores[pos_b][j]) - float(scores[pos_a][i])
        if absolute:
            d = abs(d)
        if min_sep is not None and d < float(min_sep):
            continue
        if max_sep is not None and d > float(max_sep):
            continue
        out.append(tuple(int(x) for x in t))
    return out


def filter_by_score_difference(
    tuples: Sequence[IndexTuple],
    *,
    scores: Sequence[Mapping[int, float]],
    pos_a: int = 0,
    pos_b: int = 1,
    min_diff: float | None = None,
    max_diff: float | None = None,
) -> IndexTuples:
    """Filter index tuples by signed score difference between two positions.

    This filter keeps tuples based on the signed difference between two tuple
    positions. It is useful for directional selections such as "position pos_b
    is behind pos_a" when the score encodes an ordering variable.

    For each tuple `t`, compute the signed difference between the scores at
    `t[pos_a]` and `t[pos_b]`. The tuple is kept when the difference falls within
    the requested bounds (ignoring bounds set to `None`).

    Args:
        tuples: Sequence of index tuples to be filtered.
        scores: Sequence of per-position score mappings.
        pos_a: First tuple position in the difference.
        pos_b: Second tuple position in the difference.
        min_diff: Optional minimum signed difference to enforce.
        max_diff: Optional maximum signed difference to enforce.

    Returns:
        List of index tuples that satisfy the difference bounds.

    Raises:
        ValueError: If pos_a or pos_b is out of range, or tuple is too short.
        KeyError: If a required index is missing from the score mappings.
    """
    _require_pos(scores, pos_a)
    _require_pos(scores, pos_b)

    out: IndexTuples = []
    for t in tuples:
        if pos_a >= len(t) or pos_b >= len(t):
            raise ValueError("Tuple length is smaller than requested positions.")
        i = int(t[pos_a])
        j = int(t[pos_b])
        if i not in scores[pos_a]:
            raise KeyError(f"Missing score for position {pos_a}, index {i}.")
        if j not in scores[pos_b]:
            raise KeyError(f"Missing score for position {pos_b}, index {j}.")

        d = float(scores[pos_b][j]) - float(scores[pos_a][i])
        if min_diff is not None and d < float(min_diff):
            continue
        if max_diff is not None and d > float(max_diff):
            continue
        out.append(tuple(int(x) for x in t))
    return out


def filter_by_score_consistency(
    tuples: Sequence[IndexTuple],
    *,
    scores1: Sequence[Mapping[int, float]],
    scores2: Sequence[Mapping[int, float]],
    pos_a: int = 0,
    pos_b: int = 1,
    relation: Relations = "lt",
) -> IndexTuples:
    """Filter tuples that satisfy the same ordering under two score definitions.

    This filter keeps a tuple only if it satisfies the requested ordering under
    two independently computed score maps (for the same tuple positions), such
    as peak and mean locations.

    For each tuple `t`, apply the requested relation to the scores at `t[pos_a]`
    and `t[pos_b]` under both `scores1` and `scores2`. The tuple is kept only
    when the relation holds for both score definitions.

    Args:
        tuples: Sequence of index tuples to be filtered.
        scores1: First set of per-position score mappings.
        scores2: Second set of per-position score mappings.
        pos_a: First tuple position in the comparison.
        pos_b: Second tuple position in the comparison.
        relation: Comparison operator applied for both score definitions.

    Returns:
        List of index tuples satisfying the ordering under both score maps.

    Raises:
        ValueError: If positions are invalid, tuple is too short, or scores
            lengths mismatch.
        KeyError: If a required index is missing from any score mapping.
    """
    _require_same_len(scores1, scores2, "scores1", "scores2")
    _require_pos(scores1, pos_a)
    _require_pos(scores1, pos_b)
    _require_pos(scores2, pos_a)
    _require_pos(scores2, pos_b)

    out: IndexTuples = []
    for t in tuples:
        if pos_a >= len(t) or pos_b >= len(t):
            raise ValueError("Tuple length is smaller than requested positions.")
        i = int(t[pos_a])
        j = int(t[pos_b])

        if i not in scores1[pos_a]:
            raise KeyError(f"Missing scores1 for position {pos_a}, index {i}.")
        if j not in scores1[pos_b]:
            raise KeyError(f"Missing scores1 for position {pos_b}, index {j}.")
        if i not in scores2[pos_a]:
            raise KeyError(f"Missing scores2 for position {pos_a}, index {i}.")
        if j not in scores2[pos_b]:
            raise KeyError(f"Missing scores2 for position {pos_b}, index {j}.")

        ok1 = _apply_comparator(float(scores1[pos_b][j]), float(scores1[pos_a][i]), relation)
        ok2 = _apply_comparator(float(scores2[pos_b][j]), float(scores2[pos_a][i]), relation)
        if ok1 and ok2:
            out.append(tuple(int(x) for x in t))
    return out


def filter_by_width_ratio(
    tuples: Sequence[IndexTuple],
    *,
    widths: Sequence[Mapping[int, float]],
    pos_a: int = 0,
    pos_b: int = 1,
    max_ratio: float = 2.0,
    symmetric: bool = True,
) -> IndexTuples:
    """Filter tuples by compatibility of per-index widths at two positions.

    This filter compares the width-like scalar at two tuple positions. It is
    useful when widths encode resolution or spread and you want to avoid mixing
    extremely different bin widths.

    For each tuple `t`, form the ratio of the width at `t[pos_b]` to the width at
    `t[pos_a]`. If `symmetric=True`, the ratio is made symmetric by using the
    larger of `r` and `1/r`. The tuple is kept when the resulting ratio does not
    exceed `max_ratio`.

    Args:
        tuples: Sequence of index tuples to be filtered.
        widths: Sequence of per-position width mappings.
        pos_a: First tuple position in the comparison.
        pos_b: Second tuple position in the comparison.
        max_ratio: Maximum allowed ratio (must be >= 1).
        symmetric: If True, enforce ratio symmetry between the two positions.

    Returns:
        List of index tuples that satisfy the width ratio constraint.

    Raises:
        ValueError: If positions are invalid, tuple is too short, or max_ratio
            < 1.
        KeyError: If a required index is missing from widths.
    """
    _require_pos(widths, pos_a)
    _require_pos(widths, pos_b)
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be >= 1.")

    out: IndexTuples = []
    for t in tuples:
        if pos_a >= len(t) or pos_b >= len(t):
            raise ValueError("Tuple length is smaller than requested positions.")
        i = int(t[pos_a])
        j = int(t[pos_b])

        if i not in widths[pos_a]:
            raise KeyError(f"Missing widths for position {pos_a}, index {i}.")
        if j not in widths[pos_b]:
            raise KeyError(f"Missing widths for position {pos_b}, index {j}.")

        wa = float(widths[pos_a][i])
        wb = float(widths[pos_b][j])
        if wa <= 0.0 or wb <= 0.0:
            continue

        r = wb / wa
        if symmetric:
            r = max(r, 1.0 / r)
        if r <= float(max_ratio):
            out.append(tuple(int(x) for x in t))
    return out


def filter_by_curve_norm_threshold(
    tuples: Sequence[IndexTuple],
    *,
    norms: Sequence[Mapping[int, float]],
    threshold: float,
    compare: Relations = "ge",
    mode: Literal["all", "any"] = "all",
) -> IndexTuples:
    """Filter tuples based on per-position curve norms.

    This filter keeps tuples based on the integrated normalization of the curve
    at each tuple position. It is useful for excluding tuples containing bins
    with negligible support.

    For each tuple `t`, compare the norm at each position `p` (using `t[p]`) to the
    given threshold. With `mode="all"`, every position must pass; with `mode="any"`,
    at least one position must pass.

    Args:
        tuples: Sequence of index tuples to be filtered.
        norms: Sequence of per-position norm mappings.
        threshold: Reference value used for filtering.
        compare: Comparison operator applied to each norm versus threshold.
        mode: Whether to require all positions ("all") or at least one ("any").

    Returns:
        List of index tuples that satisfy the norm threshold condition.

    Raises:
        ValueError: If mode is not recognized, or norms do not cover tuple
            positions.
        KeyError: If a required index is missing from norms.
    """
    if mode not in ("all", "any"):
        raise ValueError(f"Unknown mode: {mode!r}")
    thr = float(threshold)

    out: IndexTuples = []
    for t in tuples:
        if len(norms) < len(t):
            raise ValueError("norms must have at least len(tuple) mappings.")
        oks: list[bool] = []
        for p, idx in enumerate(t):
            ii = int(idx)
            if ii not in norms[p]:
                raise KeyError(f"Missing norm for position {p}, index {ii}.")
            oks.append(_apply_comparator(float(norms[p][ii]), thr, compare))
        if (mode == "all" and all(oks)) or (mode == "any" and any(oks)):
            out.append(tuple(int(x) for x in t))
    return out
