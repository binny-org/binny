"""Unit tests for ``binny.utils.normalization``."""

from __future__ import annotations

import numpy as np
import pytest

import binny.axes.bin_edges as bemod
from binny.utils.normalization import (
    cdf_from_curve,
    curve_norm_mode,
    integrate_bins,
    normalize_1d,
    normalize_edges,
    normalize_or_check_curves,
    prepare_metric_inputs,
    trapz_weights,
    weighted_quantile_from_cdf,
)


def test_normalize_1d_trapezoid_returns_unit_integral():
    """Tests that normalize_1d returns y with unit trapezoid integral."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    out = normalize_1d(x, y, method="trapezoid")
    assert out.dtype == np.float64
    assert np.isclose(np.trapezoid(out, x=x), 1.0)


def test_normalize_1d_simpson_returns_unit_integral():
    """Tests that normalize_1d returns y with unit simpson integral."""
    x = np.linspace(0.0, 1.0, 11)
    y = x**2 + 1.0
    out = normalize_1d(x, y, method="simpson")
    assert out.dtype == np.float64
    assert np.isclose(np.trapezoid(out, x=x), 1.0, rtol=2e-3, atol=1e-6)


def test_normalize_1d_rejects_unknown_method():
    """Tests that normalize_1d rejects unknown integration methods."""
    with pytest.raises(ValueError, match=r"method must be"):
        normalize_1d([0, 1], [1, 1], method="nope")  # type: ignore[arg-type]


def test_normalize_1d_rejects_non_positive_norm():
    """Tests that normalize_1d rejects non-positive normalization factor."""
    with pytest.raises(ValueError, match=r"Normalization factor must be positive"):
        normalize_1d([0, 1], [0, 0], method="trapezoid")  # type: ignore[arg-type]


def test_integrate_bins_happy_path_returns_integrals():
    """Tests that integrate_bins returns trapezoid integrals per bin."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.array([0.0, 1.0, 0.0]), 1: np.array([1.0, 1.0, 1.0])}
    out = integrate_bins(z, bins)
    assert set(out) == {0, 1}
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[1], 2.0)


def test_integrate_bins_rejects_empty():
    """Tests that integrate_bins rejects empty bins mapping."""
    with pytest.raises(ValueError, match=r"bins must not be empty"):
        integrate_bins([0, 1], {})  # type: ignore[arg-type]


def test_integrate_bins_annotates_invalid_bin_index():
    """Tests that integrate_bins annotates the offending bin index in errors."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {7: np.array([1.0, np.nan, 1.0])}
    with pytest.raises(ValueError, match=r"Invalid bin 7:"):
        integrate_bins(z, bins)


def test_cdf_from_curve_returns_cdf_and_norm():
    """Tests that cdf_from_curve returns cdf starting at 0 and correct norm."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([0.0, 1.0, 0.0])
    cdf, norm = cdf_from_curve(z, nz)
    assert cdf.dtype == np.float64
    assert cdf[0] == 0.0
    assert np.isclose(norm, 1.0)
    assert np.isclose(cdf[-1], norm)


def test_cdf_from_curve_rejects_negative_values():
    """Tests that cdf_from_curve rejects curves with negative values."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match=r"must be nonnegative"):
        cdf_from_curve(z, nz)


def test_cdf_from_curve_rejects_non_positive_total_mass():
    """Tests that cdf_from_curve rejects curves with non-positive total mass."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.zeros_like(z)
    with pytest.raises(ValueError, match=r"Total weight must be positive"):
        cdf_from_curve(z, nz)


def test_weighted_quantile_from_cdf_endpoints():
    """Tests that weighted_quantile_from_cdf returns endpoints at q=0 and q=1."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 0.5, 1.0])
    assert weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.0) == 0.0
    assert weighted_quantile_from_cdf(z, cdf, norm=1.0, q=1.0) == 2.0


def test_weighted_quantile_from_cdf_linear_interp():
    """Tests that weighted_quantile_from_cdf interpolates between nodes."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 0.5, 1.0])
    q = weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.25)
    assert np.isclose(q, 0.5)


def test_weighted_quantile_from_cdf_rejects_bad_inputs():
    """Tests that weighted_quantile_from_cdf rejects invalid q and norm."""
    z = np.array([0.0, 1.0])
    cdf = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"q must be between 0 and 1"):
        weighted_quantile_from_cdf(z, cdf, norm=1.0, q=1.1)
    with pytest.raises(ValueError, match=r"norm must be positive"):
        weighted_quantile_from_cdf(z, cdf, norm=0.0, q=0.5)


def test_trapz_weights_matches_np_trapezoid():
    """Tests that trapz_weights reproduces np.trapezoid via dot product."""
    z = np.array([0.0, 1.0, 3.0])
    f = np.array([2.0, 4.0, 6.0])
    w = trapz_weights(z)
    assert np.isclose(np.sum(w * f), np.trapezoid(f, x=z))


def test_trapz_weights_small_grids_return_zeros():
    """Tests that trapz_weights returns zeros for grids with fewer than 2 points."""
    z = np.array([1.0])
    w = trapz_weights(z)
    assert w.dtype == np.float64
    assert np.allclose(w, 0.0)


def test_trapz_weights_rejects_not_increasing():
    """Tests that trapz_weights rejects non-increasing grids."""
    with pytest.raises(ValueError, match=r"strictly increasing"):
        trapz_weights([0.0, 0.0, 1.0])  # type: ignore[arg-type]


def test_normalize_or_check_curves_normalizes_when_requested():
    """Tests that normalize_or_check_curves normalizes curves when normalize=True."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    out = normalize_or_check_curves(z, p, normalize=True, check_normalized=False)
    assert np.isclose(np.trapezoid(out[0], x=z), 1.0)


def test_normalize_or_check_curves_check_raises_if_not_normalized():
    """Tests that normalize_or_check_curves raises if check_normalized fails."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    with pytest.raises(ValueError, match=r"does not appear normalized"):
        normalize_or_check_curves(z, p, normalize=False, check_normalized=True)


def test_normalize_or_check_curves_warns_if_already_normalized():
    """Tests that normalize_or_check_curves warns when renormalizing unit curves."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 1.0, 0.0])}
    with pytest.warns(UserWarning, match=r"appears already normalized"):
        normalize_or_check_curves(
            z,
            p,
            normalize=True,
            check_normalized=False,
            warn_if_already_normalized=True,
        )


def test_normalize_edges_from_mapping_happy_path():
    """Tests that normalize_edges returns edges for requested indices from mapping."""
    edges = {0: (0.0, 1.0), 2: (2.0, 3.0)}
    out = normalize_edges([0, 2], edges)
    assert out == {0: (0.0, 1.0), 2: (2.0, 3.0)}


def test_normalize_edges_mapping_missing_index_raises():
    """Tests that normalize_edges raises if mapping misses a requested index."""
    edges = {0: (0.0, 1.0)}
    with pytest.raises(ValueError, match=r"missing bin index"):
        normalize_edges([0, 1], edges)


def test_normalize_edges_from_array_happy_path():
    """Tests that normalize_edges converts a strictly increasing edge array."""
    out = normalize_edges([0, 1], [0.0, 1.0, 3.0])
    assert out == {0: (0.0, 1.0), 1: (1.0, 3.0)}


def test_normalize_edges_array_rejects_out_of_range_bin():
    """Tests that normalize_edges rejects bin indices out of range for edge array."""
    with pytest.raises(ValueError, match=r"at least"):
        normalize_edges([2], [0.0, 1.0, 2.0])


def test_prepare_metric_inputs_curves_mode_none():
    """Tests that prepare_metric_inputs returns validated curves in curves mode."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    z_out, out = prepare_metric_inputs(z, p, mode="curves", curve_norm="none")
    assert z_out.dtype == np.float64
    assert out[0].shape == z.shape


def test_prepare_metric_inputs_segments_prob_has_expected_shape():
    """Tests that prepare_metric_inputs returns per-segment arrays of length n-1."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    _, out = prepare_metric_inputs(
        z,
        p,
        mode="segments_prob",
        curve_norm="normalize",
    )
    assert out[0].shape == (z.size - 1,)


def test_prepare_metric_inputs_segments_prob_sums_to_one():
    """Tests that prepare_metric_inputs segment probabilities sum to 1."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    _, out = prepare_metric_inputs(z, p, mode="segments_prob", curve_norm="normalize")
    assert np.isclose(float(np.sum(out[0])), 1.0)


def test_prepare_metric_inputs_check_mode_raises_when_not_normalized():
    """Tests that prepare_metric_inputs raises if curve_norm='check' fails."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    with pytest.raises(ValueError, match=r"does not appear normalized"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="check")


def test_prepare_metric_inputs_rejects_unknown_mode():
    """Tests that prepare_metric_inputs rejects unknown modes."""
    z = np.array([0.0, 1.0])
    p = {0: np.array([1.0, 1.0])}
    with pytest.raises(ValueError, match=r"mode must be"):
        prepare_metric_inputs(z, p, mode="nope")  # type: ignore[arg-type]


def test_curve_norm_mode_returns_expected_values():
    """Tests that curve_norm_mode returns the correct normalization mode."""
    assert (
        curve_norm_mode(
            required=False,
            assume_normalized=True,
            normalize_if_needed=True,
        )
        == "none"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=False,
            normalize_if_needed=True,
        )
        == "none"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=True,
            normalize_if_needed=True,
        )
        == "normalize"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=True,
            normalize_if_needed=False,
        )
        == "check"
    )


def test_trapz_weights_rejects_not_1d():
    """Tests that trapz_weights raises for non-1D inputs."""
    with pytest.raises(ValueError, match=r"1D"):
        trapz_weights(np.zeros((2, 2)))


def test_normalize_edges_array_rejects_bad_shape_and_nonfinite():
    """Tests that normalize_edges rejects bad edge arrays and non-finite values."""
    with pytest.raises(ValueError, match=r"1D sequence"):
        normalize_edges([0], np.array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match=r"finite"):
        normalize_edges([0], [0.0, np.nan])


def test_normalize_edges_mapping_coerces_to_float():
    """Tests that normalize_edges mapping case coerces lo/hi to floats."""
    edges = {0: ("0.0", "1.0")}
    out = normalize_edges([0], edges)
    assert out[0] == (0.0, 1.0)
    assert isinstance(out[0][0], float)
    assert isinstance(out[0][1], float)


def test_normalize_or_check_curves_raises_on_non_positive_integral():
    """Tests that normalize_or_check_curves raises for non-positive trapezoid area."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}
    with pytest.raises(ValueError, match=r"non-positive integral"):
        normalize_or_check_curves(z, p, normalize=False, check_normalized=False)


def test_normalize_or_check_curves_casts_keys_to_int():
    """Tests that normalize_or_check_curves casts bin keys to int."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0.0: np.array([0.0, 2.0, 0.0])}
    out = normalize_or_check_curves(z, p, normalize=False, check_normalized=False)
    assert set(out.keys()) == {0}


def test_weighted_quantile_from_cdf_rejects_shape_mismatch_and_non_monotone():
    """Tests that weighted_quantile_from_cdf rejects invalid array shapes/ordering."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"same nonzero length"):
        weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.5)

    z_bad = np.array([0.0, 0.0, 1.0])
    cdf_ok = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match=r"strictly increasing"):
        weighted_quantile_from_cdf(z_bad, cdf_ok, norm=1.0, q=0.5)

    z_ok = np.array([0.0, 1.0, 2.0])
    cdf_bad = np.array([0.0, 0.7, 0.6])
    with pytest.raises(ValueError, match=r"nondecreasing"):
        weighted_quantile_from_cdf(z_ok, cdf_bad, norm=1.0, q=0.5)


def test_weighted_quantile_from_cdf_handles_flat_step():
    """Tests that weighted_quantile_from_cdf returns z[j] when cdf step is flat."""
    z = np.array([0.0, 1.0, 2.0])
    # Flat segment at indices 1->2, so c0==c1 for j=2.
    cdf = np.array([0.0, 0.5, 0.5])
    q = weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.75)
    assert q == 2.0


def test_prepare_metric_inputs_rejects_bad_z():
    """Tests that prepare_metric_inputs rejects invalid z arrays."""
    p = {0: np.array([1.0, 1.0])}

    with pytest.raises(ValueError, match=r"at least two points"):
        prepare_metric_inputs([0.0], p, mode="curves")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"finite"):
        prepare_metric_inputs([0.0, np.nan], p, mode="curves")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"strictly increasing"):
        prepare_metric_inputs([0.0, 0.0], p, mode="curves")  # type: ignore[arg-type]


def test_prepare_metric_inputs_empty_bins_returns_empty_dict():
    """Tests that prepare_metric_inputs returns empty dict for empty bins mapping."""
    z = np.array([0.0, 1.0])
    z_out, out = prepare_metric_inputs(z, {}, mode="curves")
    assert np.allclose(z_out, z)
    assert out == {}


def test_prepare_metric_inputs_segments_prob_rejects_non_positive_integral():
    """Tests that prepare_metric_inputs rejects bins with non-positive integral."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}
    with pytest.raises(ValueError, match=r"non-positive or non-finite integral"):
        prepare_metric_inputs(z, p, mode="segments_prob", curve_norm="none")


def test_prepare_metric_inputs_normalize_makes_unit_integral():
    """Tests that prepare_metric_inputs curve_norm='normalize' yields unit integrals."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0]), 1: np.array([1.0, 1.0, 1.0])}
    z_out, out = prepare_metric_inputs(z, p, mode="curves", curve_norm="normalize")
    assert z_out.dtype == np.float64
    for _i, y in out.items():
        assert np.isclose(np.trapezoid(y, x=z_out), 1.0)


def test_prepare_metric_inputs_check_accepts_already_normalized():
    """Tests that prepare_metric_inputs curve_norm='check' accepts unit-integral bins."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 1.0, 0.0])}
    _, out = prepare_metric_inputs(z, p, mode="curves", curve_norm="check")
    assert np.isclose(np.trapezoid(out[0], x=z), 1.0)


def test_prepare_metric_inputs_rejects_invalid_bin_with_index_annotation():
    """Tests that prepare_metric_inputs annotates invalid bin index on failure."""
    z = np.array([0.0, 1.0, 2.0])
    p = {3: np.array([1.0, np.nan, 1.0])}
    with pytest.raises(ValueError, match=r"Invalid bin 3:"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="none")


def test_as_bins_dict_casts_keys_and_values_to_float_arrays():
    """Tests that as_bins_dict casts keys to int and values to float arrays."""
    from binny.utils.normalization import as_bins_dict

    bins = {"2": [0, 1, 2]}
    out = as_bins_dict(bins)
    assert set(out.keys()) == {2}
    assert out[2].dtype == np.float64
    assert np.allclose(out[2], np.array([0.0, 1.0, 2.0]))


def test_require_bins_uses_cached_when_bins_is_none():
    """Tests that require_bins falls back to cached bins when bins is None."""
    from binny.utils.normalization import require_bins

    cached = {0: [0, 1, 2]}
    out = require_bins(None, cached=cached)
    assert set(out.keys()) == {0}
    assert out[0].dtype == np.float64


def test_require_bins_raises_when_missing_bins_and_cached():
    """Tests that require_bins raises when neither bins nor cached are provided."""
    from binny.utils.normalization import require_bins

    with pytest.raises(ValueError, match=r"bins is not set"):
        require_bins(None, cached=None)


def test_as_float_array_raises_on_uncoercible_input():
    """Tests that as_float_array raises ValueError for inputs that cannot be coerced."""
    from binny.utils.normalization import as_float_array

    class _Bad:
        def __array__(self, dtype=None):
            raise TypeError("nope")

    with pytest.raises(ValueError, match=r"Could not convert x to a float array"):
        as_float_array(_Bad(), name="x")


def test_curve_norm_mode_required_true_assume_true_normalize_false_returns_check():
    """Tests that curve_norm_mode returns 'check' when required and assume_normalized."""
    from binny.utils.normalization import curve_norm_mode

    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=True,
            normalize_if_needed=False,
        )
        == "check"
    )


def test_prepare_metric_inputs_rejects_unknown_mode_message():
    """Tests that prepare_metric_inputs raises for unknown mode string."""
    from binny.utils.normalization import prepare_metric_inputs

    z = np.array([0.0, 1.0])
    bins = {0: np.array([1.0, 1.0])}
    with pytest.raises(ValueError, match=r"mode must be 'curves' or 'segments_prob'"):
        prepare_metric_inputs(z, bins, mode="bad")  # type: ignore[arg-type]


def test_normalization_prepare_metric_inputs_rejects_unknown_mode():
    """Tests that normalization.prepare_metric_inputs rejects unknown modes."""
    from binny.utils.normalization import prepare_metric_inputs

    z = np.array([0.0, 1.0])
    bins = {0: np.array([1.0, 1.0])}

    with pytest.raises(ValueError, match=r"mode must be 'curves' or 'segments_prob'"):
        prepare_metric_inputs(z, bins, mode="bad")  # type: ignore[arg-type]


def test_normalization_prepare_metric_inputs_rejects_nonfinite_integral_when_required():
    """Tests that prepare_metric_inputs rejects non-finite integrals when needed."""
    from binny.utils.normalization import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    # y has NaN -> validate_axis_and_weights should fail, but we also want the
    # index-annotated error branch.
    bins = {7: np.array([1.0, np.nan, 1.0])}

    with pytest.raises(ValueError, match=r"Invalid bin 7:"):
        prepare_metric_inputs(z, bins, mode="curves", curve_norm="none")


def test_normalization_prepare_metric_inputs_segments_prob_requires_positive_mass():
    """Tests that segments_prob mode rejects non-positive mass curves."""
    from binny.utils.normalization import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive or non-finite integral"):
        prepare_metric_inputs(z, bins, mode="segments_prob", curve_norm="none")


def test_normalization_prepare_metric_inputs_check_rejects_not_unit_integral():
    """Tests that curve_norm='check' rejects non-unit-integral curves."""
    from binny.utils.normalization import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.array([0.0, 2.0, 0.0])}  # area=2

    with pytest.raises(ValueError, match=r"does not appear normalized"):
        prepare_metric_inputs(z, bins, mode="curves", curve_norm="check")


def test_equal_weight_edges_repeated_edges_guard_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that _equal_weight_edges raises if interpolation yields non-increasing edges."""

    def _bad_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        _, _ = x, xp
        return float(fp[0])  # force repeated interior edges

    monkeypatch.setattr(bemod.np, "interp", _bad_interp)

    x = np.linspace(0.0, 1.0, 11)
    w = np.ones_like(x)

    with pytest.raises(ValueError, match=r"Cannot construct strictly increasing bin edges"):
        _ = bemod._equal_weight_edges(x, w, 5)


def test_weighted_quantile_from_cdf_side_right_moves_past_flat_step():
    """Tests that side='right' can move past a flat CDF step."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 0.5, 0.5])
    norm = 1.0

    q_left = weighted_quantile_from_cdf(z, cdf, norm=norm, q=0.5, side="left")
    q_right = weighted_quantile_from_cdf(z, cdf, norm=norm, q=0.5, side="right")

    assert q_left == 1.0
    assert q_right == 2.0


def test_weighted_quantile_from_cdf_searchsorted_overrun_returns_last():
    """Tests that weighted_quantile_from_cdf returns last node when j >= size."""
    z = np.array([0.0, 1.0])
    cdf = np.array([0.0, 1.0])
    # If norm is slightly smaller than last CDF but target ends up larger due to float,
    # you can force the branch by giving q=1 and norm=1 but using a cdf that doesn't reach norm.
    # Better: direct overrun with inconsistent inputs.
    assert weighted_quantile_from_cdf(z, cdf, norm=2.0, q=1.0) == 1.0


def test_trapz_weights_empty_grid_returns_empty():
    """Tests that trapz_weights returns empty weights for an empty grid."""
    z = np.array([])
    w = trapz_weights(z)
    assert w.shape == (0,)
    assert w.dtype == np.float64
