"""Unit tests for ``binny.utils.pairwise_metrics``."""

import numpy as np
import pytest

from binny.utils.pairwise_metrics import (
    apply_unit,
    fill_symmetric,
    pair_cosine,
    pair_hellinger,
    pair_js,
    pair_min,
    pair_tv,
    segment_mass_probs,
)


def _simple_curves():
    z = np.array([0.0, 1.0, 2.0])
    p = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
    }
    return z, p


def test_pair_min_returns_expected_overlap():
    """Tests that pair_min returns the trapezoid integral of pointwise minimum."""
    z, p = _simple_curves()
    f = pair_min(z, p)
    assert np.isclose(f(0, 1), 1.0)
    assert np.isclose(f(0, 2), 0.0)


def test_pair_cosine_returns_one_for_identical_curves():
    """Tests that pair_cosine returns 1 for identical nonzero curves."""
    z, p = _simple_curves()
    f = pair_cosine(z, p)
    assert np.isclose(f(0, 1), 1.0)


def test_pair_cosine_returns_zero_for_zero_norm_curve():
    """Tests that pair_cosine returns 0 when one curve has zero trapezoid L2 norm."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z), 1: np.array([0.0, 1.0, 0.0])}
    f = pair_cosine(z, p)
    assert f(0, 1) == 0.0
    assert f(1, 0) == 0.0


def test_pair_js_zero_for_identical_probs():
    """Tests that pair_js returns 0 for identical probability vectors."""
    masses = {0: np.array([0.2, 0.8]), 1: np.array([0.2, 0.8])}
    f = pair_js(masses)
    assert np.isclose(f(0, 1), 0.0)


def test_pair_js_positive_for_different_probs():
    """Tests that pair_js returns a positive distance for different vectors."""
    masses = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    f = pair_js(masses)
    v = f(0, 1)
    assert 0.0 <= v <= 1.0
    assert v > 0.0


def test_pair_js_raises_for_non_probability_vector():
    """Tests that pair_js raises for inputs that are not probability vectors."""
    masses = {0: np.array([0.2, 0.2]), 1: np.array([0.5, 0.5])}
    f = pair_js(masses)
    with pytest.raises(ValueError, match=r"masses\[0\] must sum to 1"):
        f(0, 1)


def test_pair_js_raises_for_shape_mismatch():
    """Tests that pair_js raises when vectors have different shapes."""
    masses = {0: np.array([0.5, 0.5]), 1: np.array([1.0, 0.0, 0.0])}
    f = pair_js(masses)
    with pytest.raises(ValueError, match=r"must have the same shape"):
        f(0, 1)


def test_pair_hellinger_zero_for_identical_probs():
    """Tests that pair_hellinger returns 0 for identical probability vectors."""
    masses = {0: np.array([0.1, 0.9]), 1: np.array([0.1, 0.9])}
    f = pair_hellinger(masses)
    assert np.isclose(f(0, 1), 0.0)


def test_pair_tv_expected_value():
    """Tests that pair_tv returns half the L1 distance between probabilities."""
    masses = {0: np.array([0.0, 1.0]), 1: np.array([1.0, 0.0])}
    f = pair_tv(masses)
    assert np.isclose(f(0, 1), 1.0)


def test_fill_symmetric_fills_diagonal_and_mirrors():
    """Tests that fill_symmetric mirrors upper triangle to lower triangle."""

    def pv(i, j):
        return float(i + j)

    out = fill_symmetric([0, 1, 2], pv)
    assert out[0][0] == 0.0
    assert out[0][2] == out[2][0]
    assert out[1][2] == out[2][1]


def test_segment_mass_probs_returns_probability_vectors():
    """Tests that segment_mass_probs returns per-segment probabilities summing to 1."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    out = segment_mass_probs(z, p)
    assert out[0].dtype == np.float64
    assert out[0].shape == (z.size - 1,)
    assert np.isclose(np.sum(out[0]), 1.0)


def test_segment_mass_probs_raises_for_non_positive_segment_mass():
    """Tests that segment_mass_probs raises when total segment mass is non-positive."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}
    with pytest.raises(ValueError, match=r"non-positive mass on segments"):
        segment_mass_probs(z, p)


def test_apply_unit_fraction_returns_same_object():
    """Tests that apply_unit returns input unchanged when unit is fraction."""
    mat = {0: {0: 0.5, 1: 0.1}, 1: {0: 0.1, 1: 0.2}}
    out = apply_unit(mat, unit="fraction")
    assert out is mat


def test_apply_unit_percent_scales_values():
    """Tests that apply_unit converts values to percent when unit is percent."""
    mat = {0: {0: 0.5, 1: 0.1}}
    out = apply_unit(mat, unit="percent")
    assert np.isclose(out[0][0], 50.0)
    assert np.isclose(out[0][1], 10.0)


def test_apply_unit_rejects_unknown_unit():
    """Tests that apply_unit rejects unknown unit strings."""
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        apply_unit({0: {0: 1.0}}, unit="nope")  # type: ignore[arg-type]


def test_prepare_metric_inputs_check_accepts_normalized_curve() -> None:
    """Tests that prepare_metric_inputs accepts normalized curves in check mode."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    # trapezoid integral = 1.0 exactly: 0.5*(0+1)*1 + 0.5*(1+0)*1 = 1
    p = {0: np.array([0.0, 1.0, 0.0])}

    z_out, curves = prepare_metric_inputs(z, p, mode="curves", curve_norm="check")
    assert z_out.dtype == np.float64
    assert 0 in curves
    assert curves[0].dtype == np.float64


def test_prepare_metric_inputs_check_rejects_non_unit_integral() -> None:
    """Tests that prepare_metric_inputs raises if check mode integral is not 1."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    # integral = 2.0
    p = {0: np.array([0.0, 2.0, 0.0])}

    with pytest.raises(ValueError, match=r"does not appear normalized"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="check")


def test_prepare_metric_inputs_normalize_divides_by_integral() -> None:
    """Tests that prepare_metric_inputs normalizes curves to unit integral."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    # integral = 2.0; normalized curve should integrate to 1.0
    p = {0: np.array([0.0, 2.0, 0.0])}

    z_out, curves = prepare_metric_inputs(z, p, mode="curves", curve_norm="normalize")
    area = float(np.trapezoid(curves[0], x=z_out))
    assert np.isclose(area, 1.0, rtol=0.0, atol=1e-12)


def test_prepare_metric_inputs_normalize_rejects_non_positive_integral() -> None:
    """Tests that prepare_metric_inputs raises in normalize mode for non-positive area."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive integral"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="normalize")


def test_prepare_metric_inputs_check_rejects_non_positive_integral() -> None:
    """Tests that prepare_metric_inputs raises in check mode for non-positive area."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive integral"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="check")


def test_prepare_metric_inputs_segments_prob_returns_probs() -> None:
    """Tests that prepare_metric_inputs returns per-segment probabilities in segments_prob mode."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    p = {
        0: np.array([0.0, 2.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
    }

    z_out, probs = prepare_metric_inputs(z, p, mode="segments_prob", curve_norm="none")
    assert z_out.dtype == np.float64
    assert set(probs.keys()) == {0, 1}
    for k, v in probs.items():
        assert isinstance(k, int)
        assert v.dtype == np.float64
        assert v.shape == (z.size - 1,)
        assert np.isclose(np.sum(v), 1.0, rtol=0.0, atol=1e-12)


def test_prepare_metric_inputs_segments_prob_rejects_non_positive_mass() -> None:
    """Tests that prepare_metric_inputs raises if segment mass is non-positive
    in segments_prob mode."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}

    with pytest.raises(ValueError, match=r"non-positive mass on segments"):
        prepare_metric_inputs(z, p, mode="segments_prob", curve_norm="none")


def test_prepare_metric_inputs_rejects_unknown_mode() -> None:
    """Tests that prepare_metric_inputs rejects unknown mode strings."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 1.0, 0.0])}

    with pytest.raises(ValueError, match=r'mode must be "curves" or "segments_prob"'):
        prepare_metric_inputs(z, p, mode="nope")  # type: ignore[arg-type]


def test_mass_per_segment_returns_expected_values() -> None:
    """Tests that mass_per_segment returns trapezoid masses per segment."""
    from binny.utils.pairwise_metrics import mass_per_segment

    z = np.array([0.0, 1.0, 3.0])
    p = np.array([0.0, 2.0, 0.0])
    # dz = [1, 2]
    # mass = 0.5*(0+2)*1 = 1
    # mass = 0.5*(2+0)*2 = 2
    m = mass_per_segment(z, p)
    np.testing.assert_allclose(m, np.array([1.0, 2.0]), rtol=0.0, atol=1e-12)
    assert m.dtype == np.float64


def test_mass_per_segment_rejects_shape_mismatch_or_not_1d() -> None:
    """Tests that mass_per_segment raises for non-1D inputs or length mismatch."""
    from binny.utils.pairwise_metrics import mass_per_segment

    z = np.array([0.0, 1.0, 2.0])
    p_bad_len = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"must be 1D arrays of the same length"):
        mass_per_segment(z, p_bad_len)

    z2 = np.array([[0.0, 1.0, 2.0]])
    p2 = np.array([[0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match=r"must be 1D arrays of the same length"):
        mass_per_segment(z2, p2)


def test_mass_per_segment_rejects_non_increasing_grid() -> None:
    """Tests that mass_per_segment raises when z_arr is not strictly increasing."""
    from binny.utils.pairwise_metrics import mass_per_segment

    z = np.array([0.0, 1.0, 1.0])
    p = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match=r"strictly increasing"):
        mass_per_segment(z, p)


def test_pairwise_prepare_metric_inputs_rejects_unknown_mode():
    """Tests that prepare_metric_inputs rejects unknown mode strings."""
    from binny.utils.pairwise_metrics import prepare_metric_inputs

    z = np.array([0.0, 1.0])
    p = {0: np.array([1.0, 1.0])}

    with pytest.raises(ValueError, match=r'mode must be "curves" or "segments_prob"'):
        prepare_metric_inputs(z, p, mode="bad")  # type: ignore[arg-type]
