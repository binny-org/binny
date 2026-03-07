"""Unit tests for binny.nz_tomo.bin_similarity."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.between_sample_metrics import (
    _curve_norm_mode,
    _pair_cosine_between,
    _pair_hellinger_between,
    _pair_js_between,
    _pair_min_between,
    _pair_tv_between,
    _prepare_curve_inputs,
    _prepare_mass_inputs,
    _rectangular_from_pair_value,
    _validate_method,
    _validate_same_grid,
    between_bin_overlap,
    between_interval_mass_matrix,
    between_overlap_pairs,
    between_pearson_matrix,
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)


def _example_z() -> np.ndarray:
    """Tests that helper z grid creation is deterministic."""
    return np.linspace(0.0, 1.0, 5)


def _example_bins() -> dict[int, np.ndarray]:
    """Tests that helper bins are deterministic and nontrivial."""
    return {
        0: np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float),
        1: np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
        2: np.array([2.0, 1.0, 0.0, 1.0, 2.0], dtype=float),
    }


def _normalized_curve(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Tests that helper curve normalization returns unit-integral curves."""
    area = float(np.trapezoid(values, x=z))
    return np.asarray(values, dtype=float) / area


def test_validate_method_accepts_supported_methods():
    """Tests that _validate_method accepts supported methods and lowercases them."""
    assert _validate_method("min") == "min"
    assert _validate_method("COSINE") == "cosine"
    assert _validate_method("Js") == "js"
    assert _validate_method("hellinger") == "hellinger"
    assert _validate_method("tv") == "tv"


def test_validate_method_raises_on_unknown():
    """Tests that _validate_method raises on unsupported methods."""
    with pytest.raises(
        ValueError, match=r'method must be "min", "cosine", "js", "hellinger", or "tv"'
    ):
        _validate_method("pearson")


def test_curve_norm_mode_returns_normalize_only_when_required_and_requested():
    """Tests that _curve_norm_mode returns normalize only when both flags are True."""
    assert _curve_norm_mode(normalize=True, requires_norm=True) == "normalize"
    assert _curve_norm_mode(normalize=False, requires_norm=True) == "none"
    assert _curve_norm_mode(normalize=True, requires_norm=False) == "none"
    assert _curve_norm_mode(normalize=False, requires_norm=False) == "none"


def test_prepare_curve_inputs_returns_sorted_indices_and_curves():
    """Tests that _prepare_curve_inputs returns sorted indices and prepared curves."""
    z = _example_z()
    bins = {2: _example_bins()[2], 0: _example_bins()[0]}

    z_m, bin_indices, curves = _prepare_curve_inputs(
        z,
        bins,
        normalize=False,
        requires_norm=False,
        rtol=1e-3,
        atol=1e-6,
    )

    assert np.allclose(z_m, z)
    assert bin_indices == [0, 2]
    assert set(curves.keys()) == {0, 2}
    assert np.allclose(curves[0], bins[0])
    assert np.allclose(curves[2], bins[2])


def test_prepare_mass_inputs_returns_sorted_indices_and_probabilities():
    """Tests that _prepare_mass_inputs returns sorted indices and segment-mass vectors."""
    z = _example_z()
    bins = _example_bins()

    bin_indices, masses = _prepare_mass_inputs(
        z,
        bins,
        normalize=False,
        requires_norm=True,
        rtol=1e-3,
        atol=1e-6,
    )

    assert bin_indices == [0, 1, 2]
    assert set(masses.keys()) == {0, 1, 2}
    for key in bin_indices:
        assert masses[key].ndim == 1
        assert np.isclose(np.sum(masses[key]), 1.0)


def test_rectangular_from_pair_value_builds_expected_nested_mapping():
    """Tests that _rectangular_from_pair_value builds a rectangular nested mapping."""
    rows = [0, 2]
    cols = [1, 3]

    out = _rectangular_from_pair_value(rows, cols, lambda i, j: 10 * i + j)

    assert out == {
        0: {1: 1.0, 3: 3.0},
        2: {1: 21.0, 3: 23.0},
    }


def test_pair_min_between_returns_expected_overlap():
    """Tests that _pair_min_between returns the pointwise-minimum overlap integral."""
    z = _example_z()
    curves_a = {0: np.array([0.0, 1.0, 1.0, 0.0, 0.0])}
    curves_b = {5: np.array([0.0, 0.0, 1.0, 1.0, 0.0])}

    evaluator = _pair_min_between(z, curves_a, curves_b)
    expected = float(np.trapezoid(np.minimum(curves_a[0], curves_b[5]), x=z))

    assert np.isclose(evaluator(0, 5), expected)


def test_pair_cosine_between_returns_one_for_identical_curves():
    """Tests that _pair_cosine_between returns one for identical curves."""
    z = _example_z()
    curve = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)
    curves_a = {0: curve}
    curves_b = {7: curve.copy()}

    evaluator = _pair_cosine_between(z, curves_a, curves_b)
    assert np.isclose(evaluator(0, 7), 1.0)


def test_pair_cosine_between_returns_zero_when_denominator_is_zero():
    """Tests that _pair_cosine_between returns zero for zero-norm curves."""
    z = _example_z()
    curves_a = {0: np.zeros_like(z)}
    curves_b = {1: np.ones_like(z)}

    evaluator = _pair_cosine_between(z, curves_a, curves_b)
    assert evaluator(0, 1) == 0.0


def test_pair_js_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_js_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_js_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)


def test_pair_hellinger_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_hellinger_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_hellinger_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)


def test_pair_tv_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_tv_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_tv_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)


def test_validate_same_grid_accepts_identical_arrays():
    """Tests that _validate_same_grid accepts identical z grids."""
    z = _example_z()
    _validate_same_grid(z, z.copy())


def test_validate_same_grid_raises_on_mismatched_shape():
    """Tests that _validate_same_grid raises on mismatched z-grid shape."""
    z0 = np.linspace(0.0, 1.0, 5)
    z1 = np.linspace(0.0, 1.0, 6)

    with pytest.raises(ValueError, match=r"same z grid"):
        _validate_same_grid(z0, z1)


def test_validate_same_grid_raises_on_mismatched_values():
    """Tests that _validate_same_grid raises on mismatched z-grid values."""
    z0 = np.linspace(0.0, 1.0, 5)
    z1 = z0.copy()
    z1[-1] = 1.1

    with pytest.raises(ValueError, match=r"same z grid"):
        _validate_same_grid(z0, z1)


def test_bin_overlap_returns_empty_dict_for_empty_bins():
    """Tests that bin_overlap returns an empty dict for empty inputs."""
    out = bin_overlap(_example_z(), {})
    assert out == {}


@pytest.mark.parametrize("method", ["min", "cosine", "js", "hellinger", "tv"])
def test_bin_overlap_returns_square_nested_mapping_for_supported_methods(method):
    """Tests that bin_overlap returns a square nested mapping for supported methods."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method=method, decimal_places=None)

    assert set(out.keys()) == {0, 1, 2}
    for i in [0, 1, 2]:
        assert set(out[i].keys()) == {0, 1, 2}


def test_bin_overlap_min_is_symmetric_and_has_unit_diagonal_when_normalized():
    """Tests that normalized min overlap is symmetric with unit diagonal."""
    z = _example_z()
    raw = _example_bins()
    bins = {k: _normalized_curve(v, z) for k, v in raw.items()}

    out = bin_overlap(z, bins, method="min", normalize=True, decimal_places=None)

    for i in [0, 1, 2]:
        assert np.isclose(out[i][i], 1.0)
    assert np.isclose(out[0][1], out[1][0])
    assert np.isclose(out[0][2], out[2][0])


def test_bin_overlap_cosine_is_symmetric_and_has_unit_diagonal_for_nonzero_curves():
    """Tests that cosine overlap is symmetric with unit diagonal for nonzero curves."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method="cosine", decimal_places=None)

    for i in [0, 1, 2]:
        assert np.isclose(out[i][i], 1.0)
    assert np.isclose(out[0][1], out[1][0])


def test_bin_overlap_js_has_zero_diagonal_for_normalized_identical_inputs():
    """Tests that Jensen-Shannon distance has zero diagonal for identical inputs."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method="js", normalize=True, decimal_places=None)

    for i in [0, 1, 2]:
        assert np.isclose(out[i][i], 0.0)


def test_bin_overlap_hellinger_has_zero_diagonal_for_normalized_identical_inputs():
    """Tests that Hellinger distance has zero diagonal for identical inputs."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method="hellinger", normalize=True, decimal_places=None)

    for i in [0, 1, 2]:
        assert np.isclose(out[i][i], 0.0)


def test_bin_overlap_tv_has_zero_diagonal_for_normalized_identical_inputs():
    """Tests that total-variation distance has zero diagonal for identical inputs."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method="tv", normalize=True, decimal_places=None)

    for i in [0, 1, 2]:
        assert np.isclose(out[i][i], 0.0)


def test_bin_overlap_percent_unit_scales_values():
    """Tests that bin_overlap percent unit scales values by 100."""
    z = _example_z()
    bins = _example_bins()

    frac = bin_overlap(z, bins, method="cosine", unit="fraction", decimal_places=None)
    pct = bin_overlap(z, bins, method="cosine", unit="percent", decimal_places=None)

    assert np.isclose(pct[0][1], 100.0 * frac[0][1])


def test_bin_overlap_rounds_when_decimal_places_is_set():
    """Tests that bin_overlap rounds output when decimal_places is set."""
    z = _example_z()
    bins = _example_bins()

    out = bin_overlap(z, bins, method="cosine", decimal_places=2)
    assert isinstance(out[0][1], float)
    assert out[0][1] == float(np.round(out[0][1], 2))


def test_bin_overlap_raises_on_unknown_method():
    """Tests that bin_overlap raises on an unknown method."""
    with pytest.raises(
        ValueError, match=r'method must be "min", "cosine", "js", "hellinger", or "tv"'
    ):
        bin_overlap(_example_z(), _example_bins(), method="bad-method")


def test_between_bin_overlap_returns_empty_dict_when_one_input_is_empty():
    """Tests that between_bin_overlap returns an empty dict when one input is empty."""
    z = _example_z()
    bins = _example_bins()

    assert between_bin_overlap(z, {}, bins) == {}
    assert between_bin_overlap(z, bins, {}) == {}


@pytest.mark.parametrize("method", ["min", "cosine", "js", "hellinger", "tv"])
def test_between_bin_overlap_returns_rectangular_nested_mapping(method):
    """Tests that between_bin_overlap returns a rectangular nested mapping."""
    z = _example_z()
    bins_a = {0: _example_bins()[0], 2: _example_bins()[2]}
    bins_b = {5: _example_bins()[1]}

    out = between_bin_overlap(z, bins_a, bins_b, method=method, decimal_places=None)

    assert set(out.keys()) == {0, 2}
    assert set(out[0].keys()) == {5}
    assert set(out[2].keys()) == {5}


def test_between_bin_overlap_percent_unit_scales_values():
    """Tests that between_bin_overlap percent unit scales values by 100."""
    z = _example_z()
    bins_a = {0: _example_bins()[0]}
    bins_b = {1: _example_bins()[1]}

    frac = between_bin_overlap(
        z, bins_a, bins_b, method="cosine", unit="fraction", decimal_places=None
    )
    pct = between_bin_overlap(
        z, bins_a, bins_b, method="cosine", unit="percent", decimal_places=None
    )

    assert np.isclose(pct[0][1], 100.0 * frac[0][1])


def test_between_bin_overlap_min_matches_self_overlap_for_same_bin_sets():
    """Tests that between_bin_overlap matches self-overlap values for identical bin sets."""
    z = _example_z()
    bins = {0: _example_bins()[0], 1: _example_bins()[1]}

    within = bin_overlap(z, bins, method="min", decimal_places=None)
    between = between_bin_overlap(z, bins, bins, method="min", decimal_places=None)

    assert np.isclose(within[0][1], between[0][1])
    assert np.isclose(within[1][0], between[1][0])


def test_between_bin_overlap_raises_on_unknown_method():
    """Tests that between_bin_overlap raises on an unknown method."""
    z = _example_z()
    bins = _example_bins()

    with pytest.raises(
        ValueError, match=r'method must be "min", "cosine", "js", "hellinger", or "tv"'
    ):
        between_bin_overlap(z, bins, bins, method="bad-method")


def test_overlap_pairs_raises_on_invalid_direction():
    """Tests that overlap_pairs raises on an invalid direction."""
    with pytest.raises(ValueError, match=r'direction must be "high" or "low"'):
        overlap_pairs(_example_z(), _example_bins(), direction="middle")  # type: ignore[arg-type]


def test_overlap_pairs_returns_unique_off_diagonal_pairs_sorted_high():
    """Tests that overlap_pairs returns unique off-diagonal pairs."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }

    out = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=0.5,
        direction="high",
        decimal_places=None,
    )

    assert out
    assert all(i < j for i, j, _ in out)
    values = [v for _, _, v in out]
    assert values == sorted(values, reverse=True)


def test_overlap_pairs_returns_pairs_sorted_low():
    """Tests that overlap_pairs returns pairs sorted ascending for low cuts."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }

    out = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=0.5,
        direction="low",
        decimal_places=None,
    )

    values = [v for _, _, v in out]
    assert values == sorted(values)


def test_overlap_pairs_rounds_values_when_requested():
    """Tests that overlap_pairs rounds output values when decimal_places is set."""
    z = _example_z()
    bins = _example_bins()

    out = overlap_pairs(
        z,
        bins,
        method="cosine",
        unit="fraction",
        threshold=0.0,
        direction="high",
        decimal_places=2,
    )

    assert out
    for _, _, v in out:
        assert v == float(np.round(v, 2))


def test_between_overlap_pairs_raises_on_invalid_direction():
    """Tests that between_overlap_pairs raises on an invalid direction."""
    with pytest.raises(ValueError, match=r'direction must be "high" or "low"'):
        between_overlap_pairs(_example_z(), _example_bins(), _example_bins(), direction="middle")  # type: ignore[arg-type]


def test_between_overlap_pairs_returns_all_matching_pairs_sorted_high():
    """Tests that between_overlap_pairs returns all matching rectangular pairs sorted descending."""
    z = _example_z()
    bins_a = {0: np.array([1.0, 0.0, 0.0, 0.0, 0.0])}
    bins_b = {
        5: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        6: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }

    out = between_overlap_pairs(
        z,
        bins_a,
        bins_b,
        method="cosine",
        unit="fraction",
        threshold=0.0,
        direction="high",
        decimal_places=None,
    )

    assert out
    values = [v for _, _, v in out]
    assert values == sorted(values, reverse=True)


def test_between_overlap_pairs_returns_all_matching_pairs_sorted_low():
    """Tests that between_overlap_pairs returns all matching rectangular pairs sorted ascending."""
    z = _example_z()
    bins_a = {0: np.array([1.0, 0.0, 0.0, 0.0, 0.0])}
    bins_b = {
        5: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        6: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }

    out = between_overlap_pairs(
        z,
        bins_a,
        bins_b,
        method="cosine",
        unit="fraction",
        threshold=1.0,
        direction="low",
        decimal_places=None,
    )

    values = [v for _, _, v in out]
    assert values == sorted(values)


def test_leakage_matrix_returns_empty_dict_for_empty_bins():
    """Tests that leakage_matrix returns an empty dict for empty inputs."""
    out = leakage_matrix(_example_z(), {}, [0.0, 0.5, 1.0])
    assert out == {}


def test_leakage_matrix_computes_expected_structure_for_sequence_edges():
    """Tests that leakage_matrix returns the expected nested mapping structure."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    }

    out = leakage_matrix(z, bins, [0.0, 0.5, 1.0], unit="fraction", decimal_places=None)

    assert set(out.keys()) == {0, 1}
    assert set(out[0].keys()) == {0, 1}
    assert set(out[1].keys()) == {0, 1}


def test_leakage_matrix_accepts_mapping_edges():
    """Tests that leakage_matrix accepts mapping-style bin edges."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    }
    edges = {0: (0.0, 0.5), 1: (0.5, 1.0)}

    out = leakage_matrix(z, bins, edges, unit="fraction", decimal_places=None)

    assert set(out.keys()) == {0, 1}
    assert set(out[0].keys()) == {0, 1}


def test_leakage_matrix_percent_unit_scales_values():
    """Tests that leakage_matrix percent unit scales values by 100."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    }

    frac = leakage_matrix(z, bins, [0.0, 0.5, 1.0], unit="fraction", decimal_places=None)
    pct = leakage_matrix(z, bins, [0.0, 0.5, 1.0], unit="percent", decimal_places=None)

    assert np.isclose(pct[0][0], 100.0 * frac[0][0])


def test_leakage_matrix_raises_on_invalid_unit():
    """Tests that leakage_matrix raises on an invalid unit."""
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        leakage_matrix(_example_z(), _example_bins(), [0.0, 0.5, 1.0], unit="bad")  # type: ignore[arg-type]


def test_leakage_matrix_raises_on_nonpositive_total_mass():
    """Tests that leakage_matrix raises when a bin has non-positive total mass."""
    z = _example_z()
    bins = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive total mass"):
        leakage_matrix(z, bins, [0.0, 0.5, 1.0])


def test_leakage_matrix_raises_on_invalid_edges():
    """Tests that leakage_matrix raises when an interval has hi <= lo."""
    z = _example_z()
    bins = {0: np.ones_like(z)}
    edges = {0: (0.5, 0.5)}

    with pytest.raises(ValueError, match=r"must satisfy hi > lo"):
        leakage_matrix(z, bins, edges)


def test_between_interval_mass_matrix_returns_empty_dict_for_empty_bins():
    """Tests that between_interval_mass_matrix returns an empty dict for empty inputs."""
    out = between_interval_mass_matrix(_example_z(), {}, [0.0, 0.5, 1.0])
    assert out == {}


def test_between_interval_mass_matrix_returns_expected_structure_for_sequence_edges():
    """Tests that between_interval_mass_matrix returns the expected nested mapping structure."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    }

    out = between_interval_mass_matrix(
        z,
        bins,
        [0.0, 0.5, 1.0],
        unit="fraction",
        decimal_places=None,
    )

    assert set(out.keys()) == {0, 2}
    assert set(out[0].keys()) == {0, 1}
    assert set(out[2].keys()) == {0, 1}


def test_between_interval_mass_matrix_accepts_mapping_edges():
    """Tests that between_interval_mass_matrix accepts mapping-style target edges."""
    z = _example_z()
    bins = {
        0: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    }
    edges = {5: (0.0, 0.5), 7: (0.5, 1.0)}

    out = between_interval_mass_matrix(z, bins, edges, unit="fraction", decimal_places=None)

    assert set(out.keys()) == {0, 2}
    assert set(out[0].keys()) == {5, 7}


def test_between_interval_mass_matrix_raises_on_invalid_unit():
    """Tests that between_interval_mass_matrix raises on an invalid unit."""
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        between_interval_mass_matrix(_example_z(), _example_bins(), [0.0, 0.5, 1.0], unit="bad")  # type: ignore[arg-type]


def test_between_interval_mass_matrix_raises_on_invalid_target_edges_shape():
    """Tests that between_interval_mass_matrix raises when target_edges has no intervals."""
    with pytest.raises(ValueError, match=r"must define at least one interval"):
        between_interval_mass_matrix(_example_z(), _example_bins(), [0.0])


def test_between_interval_mass_matrix_raises_on_nonpositive_total_mass():
    """Tests that between_interval_mass_matrix raises when a bin has non-positive total mass."""
    z = _example_z()
    bins = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive total mass"):
        between_interval_mass_matrix(z, bins, [0.0, 0.5, 1.0])


def test_between_interval_mass_matrix_raises_on_invalid_target_edges():
    """Tests that between_interval_mass_matrix raises when an interval has hi <= lo."""
    z = _example_z()
    bins = {0: np.ones_like(z)}
    edges = {0: (0.5, 0.5)}

    with pytest.raises(ValueError, match=r"must satisfy hi > lo"):
        between_interval_mass_matrix(z, bins, edges)


def test_pearson_matrix_returns_empty_dict_for_empty_bins():
    """Tests that pearson_matrix returns an empty dict for empty inputs."""
    out = pearson_matrix(_example_z(), {})
    assert out == {}


def test_pearson_matrix_is_symmetric_with_unit_diagonal_for_nonconstant_curves():
    """Tests that pearson_matrix is symmetric with unit diagonal for nonconstant curves."""
    z = _example_z()
    bins = {
        0: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        1: np.array([2.0, 1.0, 0.0, 1.0, 2.0]),
    }

    out = pearson_matrix(z, bins, decimal_places=None)

    assert np.isclose(out[0][0], 1.0)
    assert np.isclose(out[1][1], 1.0)
    assert np.isclose(out[0][1], out[1][0])


def test_pearson_matrix_returns_zero_when_one_curve_has_zero_std():
    """Tests that pearson_matrix returns zero when a curve has zero standard deviation."""
    z = _example_z()
    bins = {
        0: np.ones_like(z),
        1: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
    }

    out = pearson_matrix(z, bins, decimal_places=None)
    assert out[0][1] == 0.0
    assert out[1][0] == 0.0


def test_pearson_matrix_normalize_option_runs():
    """Tests that pearson_matrix accepts normalize=True and returns a nested mapping."""
    z = _example_z()
    bins = _example_bins()

    out = pearson_matrix(z, bins, normalize=True, decimal_places=None)

    assert set(out.keys()) == {0, 1, 2}
    assert set(out[0].keys()) == {0, 1, 2}


def test_between_pearson_matrix_returns_empty_dict_when_one_input_is_empty():
    """Tests that between_pearson_matrix returns an empty dict when one input is empty."""
    z = _example_z()
    bins = _example_bins()

    assert between_pearson_matrix(z, {}, bins) == {}
    assert between_pearson_matrix(z, bins, {}) == {}


def test_between_pearson_matrix_returns_rectangular_mapping():
    """Tests that between_pearson_matrix returns a rectangular nested mapping."""
    z = _example_z()
    bins_a = {
        0: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        2: np.array([2.0, 1.0, 0.0, 1.0, 2.0]),
    }
    bins_b = {
        5: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
    }

    out = between_pearson_matrix(z, bins_a, bins_b, decimal_places=None)

    assert set(out.keys()) == {0, 2}
    assert set(out[0].keys()) == {5}
    assert set(out[2].keys()) == {5}


def test_between_pearson_matrix_returns_zero_when_one_curve_has_zero_std():
    """Tests that between_pearson_matrix returns zero when a curve has zero standard deviation."""
    z = _example_z()
    bins_a = {0: np.ones_like(z)}
    bins_b = {1: np.array([0.0, 1.0, 2.0, 1.0, 0.0])}

    out = between_pearson_matrix(z, bins_a, bins_b, decimal_places=None)
    assert out[0][1] == 0.0


def test_between_pearson_matrix_normalize_option_runs():
    """Tests that between_pearson_matrix accepts normalize=True and returns a nested mapping."""
    z = _example_z()
    bins_a = {0: _example_bins()[0]}
    bins_b = {1: _example_bins()[1]}

    out = between_pearson_matrix(z, bins_a, bins_b, normalize=True, decimal_places=None)

    assert set(out.keys()) == {0}
    assert set(out[0].keys()) == {1}
