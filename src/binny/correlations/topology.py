"""Index-pair and index-tuple topology helpers.

This module provides physics-agnostic utilities for building collections of
index correlations and index tuples from one or more sets of integer keys. The focus
is on describing common combinatorial topologies that arise when iterating
over correlations, blocks, or multi-index objects.

Pairs are described using two indices i and j, while tuples are described
using indices i_1, i_2, ..., i_n. All outputs are ordered and preserve the
order of the input keys. Some topologies treat index positions symmetrically,
while others assign a distinct key set to each position.

Supported topologies include full Cartesian products, diagonal selections,
and restricted symmetric subsets such as triangular or nondecreasing
collections.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations_with_replacement, product

Pair = tuple[int, int]
Pairs = list[Pair]

IndexTuple = tuple[int, ...]
IndexTuples = list[IndexTuple]

__all__ = [
    "pairs_all",
    "pairs_upper_triangle",
    "pairs_lower_triangle",
    "pairs_diagonal",
    "pairs_off_diagonal",
    "pairs_cartesian",
    "tuples_all",
    "tuples_nondecreasing",
    "tuples_diagonal",
    "tuples_cartesian",
]


def pairs_all(keys: Sequence[int]) -> Pairs:
    """Construct all ordered index correlations from a single key set.

    This topology enumerates every possible combination of two indices drawn
    from the same set. The order of indices matters, so both (i, j) and (j, i)
    are included whenever i and j are distinct.

    Formally, this returns all correlations (i, j) such that i and j are elements of
    the provided key sequence.

    Args:
        keys: Sequence of integer keys.

    Returns:
        List of ordered correlations (i, j).
    """
    ks = list(keys)
    return [(i, j) for i in ks for j in ks]


def pairs_upper_triangle(keys: Sequence[int]) -> Pairs:
    """Construct ordered correlations forming an upper-triangular subset.

    This topology selects a symmetric subset of correlations by keeping only those
    whose first index does not come after the second under the given key
    ordering. Each unordered pair appears exactly once.

    In index terms, this returns all correlations (i, j) such that i and j are in the
    key set and i precedes or coincides with j in the key order.

    Args:
        keys: Sequence of integer keys defining the ordering.

    Returns:
        List of ordered correlations (i, j) with i not after j.
    """
    ks = list(keys)
    out: Pairs = []
    for a, i in enumerate(ks):
        for j in ks[a:]:
            out.append((i, j))
    return out


def pairs_lower_triangle(keys: Sequence[int]) -> Pairs:
    """Construct ordered correlations forming a lower-triangular subset.

    This topology is the complement of the upper-triangular selection and
    retains only those correlations where the first index does not come before the
    second under the given key ordering.

    In index terms, this returns all correlations (i, j) such that i and j are in the
    key set and i follows or coincides with j in the key order.

    Args:
        keys: Sequence of integer keys defining the ordering.

    Returns:
        List of ordered correlations (i, j) with i not before j.
    """
    ks = list(keys)
    out: Pairs = []
    for a, i in enumerate(ks):
        for j in ks[: a + 1]:
            out.append((i, j))
    return out


def pairs_diagonal(keys: Sequence[int]) -> Pairs:
    """Construct diagonal index correlations.

    This topology keeps only self-correlations, where both indices refer to the same
    key. It is commonly used when selecting auto-correlations or identity
    blocks.

    In index notation, this returns all correlations (i, i) for i in the key set.

    Args:
        keys: Sequence of integer keys.

    Returns:
        List of diagonal correlations (i, i).
    """
    ks = list(keys)
    return [(i, i) for i in ks]


def pairs_off_diagonal(keys: Sequence[int]) -> Pairs:
    """Construct all ordered off-diagonal index correlations.

    This topology includes every ordered pair except self-correlations. It is useful
    when diagonal contributions must be excluded while preserving order.

    In index notation, this returns all correlations (i, j) such that i and j are in
    the key set and i differs from j.

    Args:
        keys: Sequence of integer keys.

    Returns:
        List of ordered off-diagonal correlations (i, j).
    """
    ks = list(keys)
    return [(i, j) for i in ks for j in ks if i != j]


def pairs_cartesian(keys_a: Sequence[int], keys_b: Sequence[int]) -> Pairs:
    """Construct ordered correlations from two distinct key sets.

    This topology forms correlations by drawing the first index from one set and the
    second index from another. The two positions are not interchangeable, and
    no symmetry is implied.

    In index notation, this returns all correlations (i, j) such that i is drawn from
    the first key set and j is drawn from the second.

    Args:
        keys_a: Sequence of integer keys for the first index.
        keys_b: Sequence of integer keys for the second index.

    Returns:
        List of ordered correlations (i, j).
    """
    ka = list(keys_a)
    kb = list(keys_b)
    return [(i, j) for i in ka for j in kb]


def tuples_all(keys: Sequence[int], n: int) -> IndexTuples:
    """Construct all ordered index tuples of fixed length.

    This topology enumerates every possible ordered tuple of length n drawn
    from a single key set. It is the direct higher-order analogue of forming
    all ordered index correlations.

    In index notation, this returns all tuples (i_1, i_2, ..., i_n) where each
    entry is drawn from the key set.

    Args:
        keys: Sequence of integer keys.
        n: Tuple length.

    Returns:
        List of ordered n-tuples.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    ks = list(keys)
    return [tuple(t) for t in product(ks, repeat=n)]


def tuples_nondecreasing(keys: Sequence[int], n: int) -> IndexTuples:
    """Construct nondecreasing index tuples of fixed length.

    This topology selects a symmetric subset of ordered tuples by requiring
    that indices do not decrease under the given key ordering. Each unordered
    combination appears exactly once.

    In index notation, this returns all tuples (i_1, i_2, ..., i_n) such that
    i_1 does not come after i_2, and so on through i_n.

    Args:
        keys: Sequence of integer keys defining the ordering.
        n: Tuple length.

    Returns:
        List of ordered n-tuples with nondecreasing indices.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    ks = list(keys)
    return [tuple(t) for t in combinations_with_replacement(ks, r=n)]


def tuples_diagonal(keys: Sequence[int], n: int) -> IndexTuples:
    """Construct diagonal index tuples of fixed length.

    This topology keeps only tuples where all positions contain the same key.
    It is useful for selecting purely diagonal or self-coupled contributions
    in higher-order structures.

    In index notation, this returns all tuples (i, i, ..., i) of length n for
    i in the key set.

    Args:
        keys: Sequence of integer keys.
        n: Tuple length.

    Returns:
        List of diagonal n-tuples.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    ks = list(keys)
    return [tuple([i] * n) for i in ks]


def tuples_cartesian(keys_list: Sequence[Sequence[int]]) -> IndexTuples:
    """Construct ordered tuples from multiple key sequences.

    This topology forms tuples by drawing one index from each provided key
    sequence. Each tuple position has its own independent key domain, and no
    symmetry between positions is assumed.

    In index notation, given key sequences K_1, K_2, ..., K_n, this returns
    all tuples (i_1, i_2, ..., i_n) with i_t drawn from K_t for t = 1..n, where
    n = len(keys_list).

    Args:
        keys_list: Sequence of key sequences, one per tuple position.

    Returns:
        List of ordered tuples formed from the Cartesian product.

    Raises:
        ValueError: If keys_list is empty.
    """
    if not keys_list:
        raise ValueError("keys_list must be non-empty.")
    seqs = [list(s) for s in keys_list]
    return [tuple(t) for t in product(*seqs)]
