"""Unit tests for ``binny.ztomo.bin_similarity`` module."""

import numpy as np
import pytest

from binny.ztomo.bin_similarity import (
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)


def _z_and_bins():
    """Creates a simple z grid and set of bins for testing."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
    }
    return z, bins


def _mass_probs_bins():
    """Creates bins whose segment-mass probabilities are disjoint."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
    }
    return z, bins


def test_bin_overlap_empty_bins_returns_empty():
    """Tests that bin_overlap returns empty dict for empty bins."""
    out = bin_overlap([0.0, 1.0], {})
    assert out == {}


def test_bin_overlap_unknown_method_raises():
    """Tests that bin_overlap raises for an unknown method."""
    z, bins = _z_and_bins()
    with pytest.raises(ValueError, match=r'method must be "min"'):
        bin_overlap(z, bins, method="nope")


def test_bin_overlap_min_fraction_has_one_diagonal_for_normalized():
    """Tests that bin_overlap min gives diagonal 1 for normalized curves."""
    z, bins = _z_and_bins()
    out = bin_overlap(z, bins, method="min", unit="fraction", normalize=True)
    assert np.isclose(out[0][0], 1.0)
    assert np.isclose(out[1][1], 1.0)
    assert np.isclose(out[2][2], 1.0)


def test_bin_overlap_min_fraction_is_symmetric():
    """Tests that bin_overlap returns a symmetric matrix for min."""
    z, bins = _z_and_bins()
    out = bin_overlap(z, bins, method="min", unit="fraction", normalize=True)
    assert np.isclose(out[0][2], out[2][0])
    assert np.isclose(out[0][1], out[1][0])


def test_bin_overlap_min_offdiag_matches_expectation():
    """Tests that bin_overlap min returns expected off-diagonal values."""
    z, bins = _z_and_bins()
    out = bin_overlap(z, bins, method="min", unit="fraction", normalize=True)
    assert np.isclose(out[0][1], 1.0)
    assert np.isclose(out[0][2], 0.0)


def test_bin_overlap_min_percent_scales_values():
    """Tests that bin_overlap percent unit scales values by 100."""
    z, bins = _z_and_bins()
    out = bin_overlap(z, bins, method="min", unit="percent", normalize=True)
    assert np.isclose(out[0][0], 100.0)
    assert np.isclose(out[0][2], 0.0)


def test_bin_overlap_cosine_identical_curves_is_one():
    """Tests that bin_overlap cosine returns 1 for identical curves."""
    z, bins = _z_and_bins()
    out = bin_overlap(z, bins, method="cosine", unit="fraction")
    assert np.isclose(out[0][1], 1.0)


def test_bin_overlap_cosine_zero_norm_gives_zero():
    """Tests that bin_overlap cosine returns 0 when a curve has zero norm."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.zeros_like(z), 1: np.array([0.0, 1.0, 0.0])}
    out = bin_overlap(z, bins, method="cosine", unit="fraction")
    assert out[0][1] == 0.0
    assert out[1][0] == 0.0


def test_bin_overlap_js_identical_probs_is_zero():
    """Tests that bin_overlap js returns 0 for identical segment probabilities."""
    z, bins = _mass_probs_bins()
    bins2 = {0: bins[0], 1: bins[0]}
    out = bin_overlap(z, bins2, method="js", unit="fraction", normalize=True)
    assert np.isclose(out[0][1], 0.0)


def test_bin_overlap_tv_opposites_is_one():
    """Tests that bin_overlap tv returns 1 for disjoint segment probabilities."""
    z, bins = _mass_probs_bins()
    out = bin_overlap(z, bins, method="tv", unit="fraction", normalize=True)
    assert np.isclose(out[0][1], 1.0)


def test_bin_overlap_hellinger_opposites_is_one():
    """Tests that bin_overlap hellinger returns 1 for disjoint probabilities."""
    z, bins = _mass_probs_bins()
    out = bin_overlap(z, bins, method="hellinger", unit="fraction", normalize=True)
    assert np.isclose(out[0][1], 1.0)


def test_overlap_pairs_rejects_bad_direction():
    """Tests that overlap_pairs rejects invalid direction values."""
    z, bins = _z_and_bins()
    with pytest.raises(ValueError, match=r'direction must be "high" or "low"'):
        overlap_pairs(z, bins, direction="nope")  # type: ignore[arg-type]


def test_overlap_pairs_high_threshold_returns_sorted_pairs():
    """Tests that overlap_pairs returns sorted pairs for direction='high'."""
    z, bins = _z_and_bins()
    out = overlap_pairs(
        z,
        bins,
        threshold=50.0,
        unit="percent",
        method="min",
        direction="high",
        normalize=True,
    )
    assert out[0][:2] == (0, 1)
    assert out[0][2] >= out[-1][2]


def test_overlap_pairs_low_threshold_returns_sorted_pairs():
    """Tests that overlap_pairs returns sorted pairs for direction='low'."""
    z, bins = _z_and_bins()
    out = overlap_pairs(
        z,
        bins,
        threshold=0.0,
        unit="fraction",
        method="min",
        direction="low",
        normalize=True,
    )
    assert out[0][:2] == (0, 2)
    assert out[0][2] <= out[-1][2]


def test_leakage_matrix_empty_bins_returns_empty():
    """Tests that leakage_matrix returns empty dict for empty bins."""
    out = leakage_matrix([0.0, 1.0], {}, [0.0, 1.0])
    assert out == {}


def test_leakage_matrix_rejects_bad_unit():
    """Tests that leakage_matrix rejects unknown units."""
    z, bins = _z_and_bins()
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        leakage_matrix(z, bins, [0.0, 1.0, 2.0], unit="nope")  # type: ignore[arg-type]


def test_leakage_matrix_from_edges_array_expected_values():
    """Tests that leakage_matrix computes expected leakage values."""
    z, bins = _z_and_bins()
    edges = [0.0, 1.0, 2.0, 3.0]
    leak = leakage_matrix(z, bins, edges, unit="fraction")

    assert np.isclose(leak[0][0], 0.5)
    assert np.isclose(leak[0][1], 0.5)
    assert np.isclose(leak[2][0], 1.0)


def test_leakage_matrix_from_edges_mapping_expected_values():
    """Tests that leakage_matrix accepts explicit mapping of edge pairs."""
    z, bins = _z_and_bins()
    edges = {0: (0.0, 1.0), 1: (1.0, 2.0), 2: (2.0, 3.0)}
    leak = leakage_matrix(z, bins, edges, unit="fraction")

    assert np.isclose(leak[0][0], 0.5)
    assert np.isclose(leak[0][1], 0.5)
    assert np.isclose(leak[2][0], 1.0)


def test_leakage_matrix_raises_on_non_positive_total_mass():
    """Tests that leakage_matrix raises when a bin has non-positive total mass."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.zeros_like(z)}
    with pytest.raises(ValueError, match=r"non-positive total mass"):
        leakage_matrix(z, bins, [0.0, 1.0], unit="fraction")


def test_leakage_matrix_raises_on_invalid_edges():
    """Tests that leakage_matrix raises if any (lo, hi) does not satisfy hi > lo."""
    z, bins = _z_and_bins()
    edges = {0: (1.0, 0.0), 1: (0.0, 2.0), 2: (0.0, 2.0)}
    with pytest.raises(ValueError, match=r"must satisfy hi > lo"):
        leakage_matrix(z, bins, edges, unit="fraction")


def test_leakage_matrix_mask_with_fewer_than_two_points_gives_zero():
    """Tests that leakage_matrix returns 0 when an edge interval has <2 grid points."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.array([0.0, 1.0, 0.0])}
    edges = {0: (0.2, 0.3)}
    leak = leakage_matrix(z, bins, edges, unit="fraction")
    assert leak[0][0] == 0.0


def test_pearson_matrix_empty_bins_returns_empty():
    """Tests that pearson_matrix returns empty dict for empty bins."""
    out = pearson_matrix([0.0, 1.0], {})
    assert out == {}


def test_pearson_matrix_is_symmetric_and_has_one_diagonal():
    """Tests that pearson_matrix is symmetric and has diagonal correlation 1."""
    z, bins = _z_and_bins()
    out = pearson_matrix(z, bins, normalize=True)
    assert np.isclose(out[0][0], 1.0)
    assert np.isclose(out[0][1], out[1][0])


def test_pearson_matrix_identical_curves_is_one_offdiag():
    """Tests that pearson_matrix returns 1 for identical curves."""
    z, bins = _z_and_bins()
    out = pearson_matrix(z, bins, normalize=True)
    assert np.isclose(out[0][1], 1.0)


def test_pearson_matrix_zero_variance_curve_gives_zero_corr():
    """Tests that pearson_matrix returns 0 when one curve has zero variance."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.ones_like(z), 1: np.array([0.0, 1.0, 0.0])}
    out = pearson_matrix(z, bins, normalize=False)
    assert out[0][1] == 0.0
    assert out[1][0] == 0.0


def test_pearson_matrix_rejects_invalid_z_grid():
    """Tests that pearson_matrix raises when z grid is not strictly increasing."""
    z = np.array([0.0, 0.0, 1.0])
    bins = {0: np.array([0.0, 1.0, 0.0])}
    with pytest.raises(ValueError, match=r"strictly increasing"):
        pearson_matrix(z, bins, normalize=False)


def test_bin_overlap_tv_scale_invariant_when_normalize_true():
    """Tests that bin_overlap tv is scale-invariant when normalize=True."""
    z, bins = _mass_probs_bins()

    # scale curves (should not matter once normalize=True)
    bins_scaled = {0: 5.0 * bins[0], 1: 2.0 * bins[1]}

    out1 = bin_overlap(z, bins, method="tv", unit="fraction", normalize=True)
    out2 = bin_overlap(z, bins_scaled, method="tv", unit="fraction", normalize=True)

    assert np.isclose(out1[0][1], out2[0][1])


@pytest.mark.parametrize("method", ["tv", "hellinger", "js"])
def test_bin_overlap_percent_unit_scales_distances(method: str):
    """Tests that bin_overlap percent unit scales distances by 100."""
    z, bins = _mass_probs_bins()

    out_frac = bin_overlap(z, bins, method=method, unit="fraction", normalize=True)
    out_pct = bin_overlap(z, bins, method=method, unit="percent", normalize=True)

    assert np.isclose(out_pct[0][1], 100.0 * out_frac[0][1])


def test_bin_overlap_rejects_bad_unit():
    """Tests that bin_overlap rejects unknown units."""
    z, bins = _z_and_bins()
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        bin_overlap(z, bins, method="min", unit="nope")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["js", "tv", "hellinger"])
def test_bin_overlap_distance_diagonal_is_zero(method: str):
    """Tests that bin_overlap distance methods return zero on the diagonal."""
    z, bins = _mass_probs_bins()
    out = bin_overlap(
        z, {0: bins[0], 1: bins[0]}, method=method, unit="fraction", normalize=True
    )
    assert np.isclose(out[0][0], 0.0)
    assert np.isclose(out[1][1], 0.0)


def test_overlap_pairs_low_direction_with_tv_selects_identicals():
    """Tests that overlap_pairs with direction='low' selects identical bins first."""
    z, bins = _mass_probs_bins()
    bins2 = {0: bins[0], 1: bins[0]}

    out = overlap_pairs(
        z,
        bins2,
        threshold=0.0,
        unit="fraction",
        method="tv",
        direction="low",
        normalize=True,
    )
    assert out == [(0, 1, 0.0)]


def test_overlap_pairs_high_direction_with_tv_orders_by_distance():
    """Tests that overlap_pairs with direction='high' orders by distance."""
    z, bins = _mass_probs_bins()
    bins3 = {0: bins[0], 1: bins[1], 2: bins[0]}

    out = overlap_pairs(
        z,
        bins3,
        threshold=0.0,
        unit="fraction",
        method="tv",
        direction="high",
        normalize=True,
    )

    assert out[-1][:2] == (0, 2)


def test_leakage_matrix_identity_case_diagonal_is_one():
    """Tests that leakage_matrix returns identity matrix in ideal case."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
    }
    edges = {0: (0.0, 1.0), 1: (1.0, 2.0)}

    leak = leakage_matrix(z, bins, edges, unit="fraction")

    assert np.isclose(leak[0][0], 1.0)
    assert np.isclose(leak[0][1], 0.0)
    assert np.isclose(leak[1][1], 1.0)
    assert np.isclose(leak[1][0], 0.0)


def test_leakage_matrix_percent_unit_scales():
    """Tests that leakage_matrix percent unit scales values by 100."""
    z, bins = _z_and_bins()
    edges = {0: (0.0, 1.0), 1: (1.0, 2.0), 2: (2.0, 3.0)}
    leak_frac = leakage_matrix(z, bins, edges, unit="fraction")
    leak_pct = leakage_matrix(z, bins, edges, unit="percent")
    assert np.isclose(leak_pct[0][0], 100.0 * leak_frac[0][0])


def test_leakage_matrix_edges_mapping_missing_index_raises():
    """Tests that leakage_matrix raises if any bin index is missing."""
    z, bins = _z_and_bins()
    edges = {0: (0.0, 1.0), 1: (1.0, 2.0)}  # missing 2
    with pytest.raises(ValueError, match=r"missing bin index"):
        leakage_matrix(z, bins, edges, unit="fraction")


def test_pearson_matrix_normalize_true_is_scale_invariant():
    """Tests that pearson_matrix is scale-invariant when normalize=True."""
    z, bins = _z_and_bins()
    bins_scaled = {k: 10.0 * v for k, v in bins.items()}

    out1 = pearson_matrix(z, bins, normalize=True)
    out2 = pearson_matrix(z, bins_scaled, normalize=True)

    assert np.isclose(out1[0][2], out2[0][2])


def test_pearson_matrix_can_be_negative():
    """Tests that pearson_matrix can return negative correlations."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {
        0: np.array([0.0, 1.0, 2.0]),
        1: np.array([2.0, 1.0, 0.0]),
    }
    out = pearson_matrix(z, bins, normalize=False)
    assert out[0][1] < 0.0
    assert np.isclose(out[0][1], out[1][0])
