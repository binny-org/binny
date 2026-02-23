"""Unit tests for binny.correlations.topology."""

from __future__ import annotations

import pytest

from binny.correlations import topology as tp


def test_pairs_all_returns_full_ordered_product():
    """Tests that pairs_all returns all ordered pairs."""
    keys = [2, 5, 7]
    out = tp.pairs_all(keys)

    assert out == [
        (2, 2),
        (2, 5),
        (2, 7),
        (5, 2),
        (5, 5),
        (5, 7),
        (7, 2),
        (7, 5),
        (7, 7),
    ]


def test_pairs_upper_triangle_returns_each_unordered_pair_once():
    """Tests that pairs_upper_triangle returns an upper-triangular subset."""
    keys = [10, 20, 30]
    out = tp.pairs_upper_triangle(keys)

    assert out == [
        (10, 10),
        (10, 20),
        (10, 30),
        (20, 20),
        (20, 30),
        (30, 30),
    ]


def test_pairs_lower_triangle_returns_each_unordered_pair_once():
    """Tests that pairs_lower_triangle returns a lower-triangular subset."""
    keys = [10, 20, 30]
    out = tp.pairs_lower_triangle(keys)

    assert out == [
        (10, 10),
        (20, 10),
        (20, 20),
        (30, 10),
        (30, 20),
        (30, 30),
    ]


def test_pairs_diagonal_returns_only_self_pairs():
    """Tests that pairs_diagonal returns only (i, i) pairs."""
    keys = [3, 9, 12]
    out = tp.pairs_diagonal(keys)

    assert out == [(3, 3), (9, 9), (12, 12)]


def test_pairs_off_diagonal_excludes_self_pairs():
    """Tests that pairs_off_diagonal excludes (i, i) pairs."""
    keys = [1, 2, 3]
    out = tp.pairs_off_diagonal(keys)

    assert (1, 1) not in out
    assert (2, 2) not in out
    assert (3, 3) not in out
    assert len(out) == 3 * 3 - 3
    assert out[0] == (1, 2)


def test_pairs_cartesian_returns_cross_product_in_order():
    """Tests that pairs_cartesian returns the ordered cross product."""
    a = [1, 2]
    b = [10, 20, 30]
    out = tp.pairs_cartesian(a, b)

    assert out == [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 20),
        (2, 30),
    ]


def test_tuples_all_raises_when_n_lt_one():
    """Tests that tuples_all raises when n < 1."""
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        _ = tp.tuples_all([1, 2], n=0)


def test_tuples_all_returns_ordered_product():
    """Tests that tuples_all returns all ordered n-tuples."""
    keys = [1, 2]
    out = tp.tuples_all(keys, n=3)

    assert len(out) == 2**3
    assert out[0] == (1, 1, 1)
    assert out[-1] == (2, 2, 2)
    assert (1, 2, 1) in out


def test_tuples_nondecreasing_raises_when_n_lt_one():
    """Tests that tuples_nondecreasing raises when n < 1."""
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        _ = tp.tuples_nondecreasing([1, 2], n=0)


def test_tuples_nondecreasing_returns_combinations_with_replacement():
    """Tests that tuples_nondecreasing returns nondecreasing tuples."""
    keys = [1, 2, 3]
    out = tp.tuples_nondecreasing(keys, n=2)

    assert out == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 3),
    ]


def test_tuples_diagonal_raises_when_n_lt_one():
    """Tests that tuples_diagonal raises when n < 1."""
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        _ = tp.tuples_diagonal([1, 2], n=0)


def test_tuples_diagonal_returns_repeated_key_tuples():
    """Tests that tuples_diagonal returns (i, i, ..., i) tuples."""
    keys = [4, 9]
    out = tp.tuples_diagonal(keys, n=3)

    assert out == [(4, 4, 4), (9, 9, 9)]


def test_tuples_cartesian_raises_when_keys_list_empty():
    """Tests that tuples_cartesian raises when keys_list is empty."""
    with pytest.raises(ValueError, match=r"keys_list must be non-empty"):
        _ = tp.tuples_cartesian([])


def test_tuples_cartesian_returns_product_over_positions():
    """Tests that tuples_cartesian returns the Cartesian product."""
    keys_list = [[1, 2], [10], [7, 8]]
    out = tp.tuples_cartesian(keys_list)

    assert out == [
        (1, 10, 7),
        (1, 10, 8),
        (2, 10, 7),
        (2, 10, 8),
    ]
