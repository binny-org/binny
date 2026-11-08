"""Tests for within-sample and between-sample tomography comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from binny.tomography.comparison import (
    between_bin_overlap,
    between_interval_mass_matrix,
    between_overlap_pairs,
    between_pearson_matrix,
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)

METHODS = ("min", "cosine", "js", "hellinger", "tv")
DISTANCES = ("js", "hellinger", "tv")


def _example():
    """Tests that example bin distributions are constructed consistently."""
    z = np.linspace(0.0, 1.0, 5)
    bins = {
        0: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        1: np.ones(5),
        2: np.array([2.0, 1.0, 0.0, 1.0, 2.0]),
    }
    return z, bins


def _smooth_samples():
    """Tests that sample distributions use independent redshift grids."""
    z_a = np.linspace(0.0, 1.5, 151)
    z_b = np.linspace(0.0, 2.0, 401)

    def gaussian(z, mu, sigma):
        return np.exp(-0.5 * ((z - mu) / sigma) ** 2)

    bins_a = {0: gaussian(z_a, 0.5, 0.15), 2: gaussian(z_a, 0.9, 0.18)}
    bins_b = {5: gaussian(z_b, 0.6, 0.16), 7: gaussian(z_b, 1.1, 0.20)}
    return z_a, bins_a, z_b, bins_b


def _manual_union(z_a, bins_a, z_b, bins_b):
    """Tests that independent samples can be interpolated onto a union grid."""
    z = np.union1d(z_a, z_b)
    a = {i: np.interp(z, z_a, curve, left=0.0, right=0.0) for i, curve in bins_a.items()}
    b = {i: np.interp(z, z_b, curve, left=0.0, right=0.0) for i, curve in bins_b.items()}
    return z, a, b


def test_bin_overlap_empty():
    """Tests that empty bin collections return an empty overlap matrix."""
    assert bin_overlap([0.0, 1.0], {}) == {}


@pytest.mark.parametrize("method", METHODS)
def test_bin_overlap_square_symmetric(method):
    """Tests that within-sample overlap matrices are square and symmetric."""
    z, bins = _example()
    out = bin_overlap(z, bins, method=method, normalize=True, decimal_places=None)

    assert set(out) == set(bins)
    for i in bins:
        assert set(out[i]) == set(bins)
        for j in bins:
            assert out[i][j] == pytest.approx(out[j][i])


@pytest.mark.parametrize("method", ("min", "cosine"))
def test_bin_overlap_similarity_diagonal(method):
    """Tests that normalized similarity metrics have unit diagonal entries."""
    z, bins = _example()
    out = bin_overlap(z, bins, method=method, normalize=True, decimal_places=None)

    assert all(out[i][i] == pytest.approx(1.0) for i in bins)


@pytest.mark.parametrize("method", DISTANCES)
def test_bin_overlap_distance_diagonal(method):
    """Tests that distance metrics vanish for identical bin distributions."""
    z, bins = _example()
    out = bin_overlap(z, bins, method=method, normalize=True, decimal_places=None)

    assert all(out[i][i] == pytest.approx(0.0, abs=1e-12) for i in bins)


@pytest.mark.parametrize("method", METHODS)
def test_bin_overlap_percent_scaling(method):
    """Tests that percent units multiply overlap values by one hundred."""
    z, bins = _example()
    frac = bin_overlap(
        z,
        bins,
        method=method,
        normalize=True,
        unit="fraction",
        decimal_places=None,
    )
    pct = bin_overlap(
        z,
        bins,
        method=method,
        normalize=True,
        unit="percent",
        decimal_places=None,
    )

    assert pct[0][2] == pytest.approx(100.0 * frac[0][2])


def test_bin_overlap_min_known_value():
    """Tests that disjoint pointwise supports have zero minimum overlap."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
    }

    out = bin_overlap(z, bins, method="min", normalize=True)
    assert out[0][1] == pytest.approx(0.0)


def test_bin_overlap_cosine_zero_norm():
    """Tests that cosine similarity is zero for a zero-norm curve."""
    z, bins = _example()
    bins[0] = np.zeros_like(z)

    out = bin_overlap(z, bins, method="cosine")
    assert out[0][1] == 0.0


@pytest.mark.parametrize("method", DISTANCES)
def test_bin_overlap_disjoint_segment_masses(method):
    """Tests that disjoint segment-mass distributions have maximum distance."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
    }

    out = bin_overlap(
        z,
        bins,
        method=method,
        normalize=True,
        decimal_places=None,
    )
    assert out[0][1] == pytest.approx(1.0)


def test_bin_overlap_invalid_method_and_unit():
    """Tests that unsupported overlap methods and units raise errors."""
    z, bins = _example()

    with pytest.raises(ValueError, match="method must be"):
        bin_overlap(z, bins, method="unknown")

    with pytest.raises(ValueError, match="unit must be"):
        bin_overlap(z, bins, unit="unknown")


def test_bin_overlap_rounding():
    """Tests that overlap values respect the requested decimal precision."""
    z, bins = _example()
    raw = bin_overlap(z, bins, method="cosine", decimal_places=None)
    out = bin_overlap(z, bins, method="cosine", decimal_places=3)

    assert out[0][2] == float(np.round(raw[0][2], 3))


@pytest.mark.parametrize("direction", ("high", "low"))
def test_overlap_pairs_threshold_and_order(direction):
    """Tests that overlap pairs satisfy thresholds and sorting requirements."""
    z, bins = _example()
    threshold = 0.5

    out = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=threshold,
        direction=direction,
        decimal_places=None,
    )

    assert all(i < j for i, j, _ in out)
    values = [v for _, _, v in out]
    assert values == sorted(values, reverse=(direction == "high"))
    assert all((v >= threshold if direction == "high" else v <= threshold) for v in values)


def test_overlap_pairs_invalid_direction():
    """Tests that unsupported overlap-pair directions raise an error."""
    z, bins = _example()

    with pytest.raises(ValueError, match="direction must be"):
        overlap_pairs(z, bins, direction="other")


def test_overlap_pairs_rounds_only_after_threshold():
    """Tests that rounding changes pair values without changing selected pairs."""
    z, bins = _example()
    raw = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=0.0,
        decimal_places=None,
    )
    rounded = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=0.0,
        decimal_places=2,
    )

    assert [(i, j) for i, j, _ in raw] == [(i, j) for i, j, _ in rounded]
    assert [v for _, _, v in rounded] == [float(np.round(v, 2)) for _, _, v in raw]


def test_leakage_empty():
    """Tests that leakage matrices are empty for empty bin collections."""
    assert leakage_matrix([0.0, 1.0], {}, [0.0, 1.0]) == {}


def test_leakage_known_values_and_edge_formats():
    """Tests that leakage values agree for sequence and mapping bin edges."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
    }
    edges_array = [0.0, 1.0, 2.0]
    edges_map = {0: (0.0, 1.0), 1: (1.0, 2.0)}

    a = leakage_matrix(z, bins, edges_array, decimal_places=None)
    b = leakage_matrix(z, bins, edges_map, decimal_places=None)

    assert a == b
    assert a[0][0] == pytest.approx(0.5)
    assert a[0][1] == pytest.approx(0.5)
    assert a[1][0] == pytest.approx(1.0)


def test_leakage_percent_scaling():
    """Tests that percent leakage values are scaled by one hundred."""
    z, bins = _example()
    edges = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    a = leakage_matrix(z, bins, edges, unit="fraction", decimal_places=None)
    b = leakage_matrix(z, bins, edges, unit="percent", decimal_places=None)

    assert b[0][1] == pytest.approx(100.0 * a[0][1])


def test_leakage_invalid_inputs():
    """Tests that invalid units, masses, and edges raise errors."""
    z, bins = _example()

    with pytest.raises(ValueError, match="unit must be"):
        leakage_matrix(z, bins, [0.0, 0.5, 1.0], unit="bad")

    with pytest.raises(ValueError, match="non-positive total mass"):
        leakage_matrix(z, {0: np.zeros_like(z)}, [0.0, 1.0])

    with pytest.raises(ValueError, match="hi > lo"):
        leakage_matrix(z, {0: bins[0]}, {0: (0.5, 0.5)})


def test_pearson_empty():
    """Tests that Pearson matrices are empty for empty bin collections."""
    assert pearson_matrix([0.0, 1.0], {}) == {}


def test_pearson_symmetry_and_diagonal():
    """Tests that Pearson matrices are symmetric with unit nonconstant diagonals."""
    z, bins = _example()
    out = pearson_matrix(z, bins, decimal_places=None)

    for i in (0, 2):
        assert out[i][i] == pytest.approx(1.0)

    assert out[0][2] == pytest.approx(out[2][0])
    assert out[0][2] < 0.0


def test_pearson_constant_curve_zero_correlation():
    """Tests that constant curves have zero Pearson correlation."""
    z, bins = _example()
    out = pearson_matrix(z, bins, decimal_places=None)

    assert out[0][1] == 0.0


def test_pearson_scale_invariance():
    """Tests that Pearson correlations are invariant under positive rescaling."""
    z, bins = _example()
    scaled = {i: (i + 2) * curve for i, curve in bins.items()}

    a = pearson_matrix(z, bins, normalize=True, decimal_places=None)
    b = pearson_matrix(z, scaled, normalize=True, decimal_places=None)

    assert a[0][2] == pytest.approx(b[0][2])


def test_pearson_rejects_nonincreasing_grid():
    """Tests that Pearson comparisons reject nonincreasing redshift grids."""
    z, bins = _example()
    z[2] = z[1]

    with pytest.raises(ValueError, match="strictly increasing"):
        pearson_matrix(z, bins)


def test_between_overlap_empty():
    """Tests that cross-sample overlap is empty when either sample has no bins."""
    z, bins = _example()

    assert between_bin_overlap(z, {}, z, bins) == {}
    assert between_bin_overlap(z, bins, z, {}) == {}


@pytest.mark.parametrize("method", METHODS)
def test_between_overlap_rectangular(method):
    """Tests that cross-sample overlap matrices have rectangular dimensions."""
    z, bins = _example()
    a = {0: bins[0], 2: bins[2]}
    b = {5: bins[1]}

    out = between_bin_overlap(
        z,
        a,
        z,
        b,
        method=method,
        normalize=True,
        decimal_places=None,
    )

    assert set(out) == {0, 2}
    assert all(set(row) == {5} for row in out.values())


@pytest.mark.parametrize("method", METHODS)
def test_between_overlap_matches_within_on_same_grid(method):
    """Tests that cross-sample metrics reproduce within-sample values."""
    z, bins = _example()

    expected = bin_overlap(
        z,
        bins,
        method=method,
        normalize=True,
        decimal_places=None,
    )
    actual = between_bin_overlap(
        z,
        bins,
        z,
        bins,
        method=method,
        normalize=True,
        decimal_places=None,
    )

    for i in bins:
        for j in bins:
            assert actual[i][j] == pytest.approx(expected[i][j], abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_between_overlap_different_grids_matches_manual_union(method):
    """Tests that cross-sample metrics agree with manual union-grid interpolation."""
    z_a, bins_a, z_b, bins_b = _smooth_samples()
    z, a, b = _manual_union(z_a, bins_a, z_b, bins_b)

    expected = between_bin_overlap(
        z,
        a,
        z,
        b,
        method=method,
        normalize=True,
        decimal_places=None,
    )
    actual = between_bin_overlap(
        z_a,
        bins_a,
        z_b,
        bins_b,
        method=method,
        normalize=True,
        decimal_places=None,
    )

    for i in bins_a:
        for j in bins_b:
            assert actual[i][j] == pytest.approx(expected[i][j], abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_between_overlap_swap_symmetry(method):
    """Tests that swapping samples transposes cross-sample overlap values."""
    z_a, a, z_b, b = _smooth_samples()

    ab = between_bin_overlap(
        z_a,
        a,
        z_b,
        b,
        method=method,
        normalize=True,
        decimal_places=None,
    )
    ba = between_bin_overlap(
        z_b,
        b,
        z_a,
        a,
        method=method,
        normalize=True,
        decimal_places=None,
    )

    for i in a:
        for j in b:
            assert ab[i][j] == pytest.approx(ba[j][i], abs=1e-12)


def test_between_overlap_nonoverlapping_support():
    """Tests that nonoverlapping sample supports have zero minimum overlap."""
    z_a = np.linspace(0.0, 1.0, 101)
    z_b = np.linspace(2.0, 3.0, 101)
    a = {0: np.ones_like(z_a)}
    b = {5: np.ones_like(z_b)}

    out = between_bin_overlap(
        z_a,
        a,
        z_b,
        b,
        method="min",
        normalize=True,
        decimal_places=None,
    )

    assert out[0][5] == pytest.approx(0.0)


def test_between_overlap_does_not_mutate_inputs():
    """Tests that cross-sample overlap leaves input grids and curves unchanged."""
    z_a, a, z_b, b = _smooth_samples()
    z_a0, z_b0 = z_a.copy(), z_b.copy()
    a0 = {k: v.copy() for k, v in a.items()}
    b0 = {k: v.copy() for k, v in b.items()}

    between_bin_overlap(z_a, a, z_b, b, method="js", normalize=True)

    np.testing.assert_array_equal(z_a, z_a0)
    np.testing.assert_array_equal(z_b, z_b0)

    for k in a:
        np.testing.assert_array_equal(a[k], a0[k])

    for k in b:
        np.testing.assert_array_equal(b[k], b0[k])


def test_between_overlap_invalid_method():
    """Tests that cross-sample overlap rejects unsupported methods."""
    z, bins = _example()

    with pytest.raises(ValueError, match="method must be"):
        between_bin_overlap(z, bins, z, bins, method="bad")


@pytest.mark.parametrize("direction", ("high", "low"))
def test_between_pairs_threshold_sorting_and_rounding(direction):
    """Tests that cross-sample pairs preserve thresholds, sorting, and rounding."""
    z_a, a, z_b, b = _smooth_samples()
    threshold = 0.0 if direction == "high" else 1.0

    raw = between_overlap_pairs(
        z_a,
        a,
        z_b,
        b,
        method="cosine",
        unit="fraction",
        threshold=threshold,
        direction=direction,
        decimal_places=None,
    )
    rounded = between_overlap_pairs(
        z_a,
        a,
        z_b,
        b,
        method="cosine",
        unit="fraction",
        threshold=threshold,
        direction=direction,
        decimal_places=2,
    )

    assert len(raw) == len(a) * len(b)
    assert {(i, j) for i, j, _ in raw} == {(i, j) for i in a for j in b}

    values = [v for _, _, v in raw]
    assert values == sorted(values, reverse=(direction == "high"))
    assert [(i, j) for i, j, _ in raw] == [(i, j) for i, j, _ in rounded]
    assert [v for _, _, v in rounded] == [float(np.round(v, 2)) for _, _, v in raw]


def test_between_pairs_invalid_direction():
    """Tests that cross-sample pair selection rejects unsupported directions."""
    z, bins = _example()

    with pytest.raises(ValueError, match="direction must be"):
        between_overlap_pairs(z, bins, z, bins, direction="bad")


def test_between_interval_mass_empty():
    """Tests that interval-mass matrices are empty for empty bin collections."""
    assert between_interval_mass_matrix([0.0, 1.0], {}, [0.0, 1.0]) == {}


def test_between_interval_mass_rectangular_and_mapping_edges():
    """Tests that interval-mass matrices support arbitrary target interval labels."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        2: np.array([0.0, 1.0, 0.0]),
        4: np.array([1.0, 0.0, 0.0]),
    }
    edges = {5: (0.0, 1.0), 7: (1.0, 2.0)}

    out = between_interval_mass_matrix(
        z,
        bins,
        edges,
        decimal_places=None,
    )

    assert set(out) == {2, 4}
    assert set(out[2]) == {5, 7}
    assert out[2][5] == pytest.approx(0.5)
    assert out[2][7] == pytest.approx(0.5)
    assert out[4][5] == pytest.approx(1.0)


def test_between_interval_mass_matches_leakage_on_matching_edges():
    """Tests that interval-mass matrices reproduce leakage values."""
    z, bins = _example()
    edges = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    a = leakage_matrix(z, bins, edges, decimal_places=None)
    b = between_interval_mass_matrix(z, bins, edges, decimal_places=None)

    assert a == b


def test_between_interval_mass_invalid_inputs():
    """Tests that interval-mass comparisons reject invalid units, masses, and edges."""
    z, bins = _example()

    with pytest.raises(ValueError, match="unit must be"):
        between_interval_mass_matrix(z, bins, [0.0, 1.0], unit="bad")

    with pytest.raises(ValueError, match="at least one interval"):
        between_interval_mass_matrix(z, bins, [0.0])

    with pytest.raises(ValueError, match="non-positive total mass"):
        between_interval_mass_matrix(
            z,
            {0: np.zeros_like(z)},
            [0.0, 1.0],
        )

    with pytest.raises(ValueError, match="hi > lo"):
        between_interval_mass_matrix(z, bins, {0: (0.5, 0.5)})


def test_between_pearson_empty():
    """Tests that cross-sample Pearson matrices are empty for empty samples."""
    z, bins = _example()

    assert between_pearson_matrix(z, {}, z, bins) == {}
    assert between_pearson_matrix(z, bins, z, {}) == {}


def test_between_pearson_rectangular():
    """Tests that cross-sample Pearson matrices have rectangular dimensions."""
    z, bins = _example()

    out = between_pearson_matrix(
        z,
        {0: bins[0], 2: bins[2]},
        z,
        {5: bins[0]},
        decimal_places=None,
    )

    assert set(out) == {0, 2}
    assert set(out[0]) == {5}
    assert out[0][5] == pytest.approx(1.0)


def test_between_pearson_matches_within_on_same_grid():
    """Tests that cross-sample Pearson values match within-sample correlations."""
    z, bins = _example()

    a = pearson_matrix(z, bins, decimal_places=None)
    b = between_pearson_matrix(z, bins, z, bins, decimal_places=None)

    for i in bins:
        for j in bins:
            assert b[i][j] == pytest.approx(a[i][j], abs=1e-12)


def test_between_pearson_different_grids_matches_manual_union():
    """Tests that cross-sample Pearson values agree with union-grid interpolation."""
    z_a, a, z_b, b = _smooth_samples()
    z, aligned_a, aligned_b = _manual_union(z_a, a, z_b, b)

    expected = between_pearson_matrix(
        z,
        aligned_a,
        z,
        aligned_b,
        decimal_places=None,
    )
    actual = between_pearson_matrix(
        z_a,
        a,
        z_b,
        b,
        decimal_places=None,
    )

    for i in a:
        for j in b:
            assert actual[i][j] == pytest.approx(expected[i][j], abs=1e-12)


def test_between_pearson_swap_symmetry():
    """Tests that swapping samples transposes cross-sample Pearson values."""
    z_a, a, z_b, b = _smooth_samples()

    ab = between_pearson_matrix(z_a, a, z_b, b, decimal_places=None)
    ba = between_pearson_matrix(z_b, b, z_a, a, decimal_places=None)

    for i in a:
        for j in b:
            assert ab[i][j] == pytest.approx(ba[j][i], abs=1e-12)


def test_between_pearson_constant_curve():
    """Tests that constant curves produce zero cross-sample Pearson correlation."""
    z, bins = _example()

    out = between_pearson_matrix(
        z,
        {0: np.ones_like(z)},
        z,
        {5: bins[0]},
        decimal_places=None,
    )

    assert out[0][5] == 0.0


def test_between_pearson_normalize_scale_invariant():
    """Tests that normalized cross-sample Pearson values are scale invariant."""
    z_a, a, z_b, b = _smooth_samples()
    scaled_a = {i: 3.0 * v for i, v in a.items()}
    scaled_b = {i: 7.0 * v for i, v in b.items()}

    x = between_pearson_matrix(
        z_a,
        a,
        z_b,
        b,
        normalize=True,
        decimal_places=None,
    )
    y = between_pearson_matrix(
        z_a,
        scaled_a,
        z_b,
        scaled_b,
        normalize=True,
        decimal_places=None,
    )

    for i in a:
        for j in b:
            assert x[i][j] == pytest.approx(y[i][j], abs=1e-12)
