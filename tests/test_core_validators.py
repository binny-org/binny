"""Unit tests for binny.core.validators."""

import numpy as np
import pytest

from binny.utils.validators import (
    resolve_binning_method,
    validate_axis_and_weights,
    validate_interval,
    validate_mixed_segments,
    validate_n_bins,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("equidistant", "equidistant"),
        (" eq ", "equidistant"),
        ("LINEAR", "equidistant"),
        ("log", "log"),
        ("log_edges", "log"),
        ("equal_number", "equal_number"),
        ("equipop", "equal_number"),
        ("en", "equal_number"),
        ("equal_information", "equal_information"),
        ("info", "equal_information"),
        ("equidistant_chi", "equidistant_chi"),
        ("chi", "equidistant_chi"),
        ("geometric", "geometric"),
        ("geom", "geometric"),
        ("geometric_edges_n", "geometric"),
    ],
)
def test_resolve_binning_method_aliases(name, expected):
    """Tests that various binning method names and aliases resolve correctly."""
    assert resolve_binning_method(name) == expected


def test_resolve_binning_method_unknown_raises():
    """Tests that an unknown binning method raises a ValueError."""
    with pytest.raises(ValueError, match=r"Unknown binning method"):
        resolve_binning_method("not_a_method")


@pytest.mark.parametrize("n_bins", [1, 2, 10, 123])
def test_validate_n_bins_accepts_positive_int(n_bins):
    """Tests that validate_n_bins accepts valid positive integers."""
    validate_n_bins(n_bins)


@pytest.mark.parametrize("bad", [0, -1, -999])
def test_validate_n_bins_rejects_non_positive(bad):
    """Tests that validate_n_bins rejects non-positive integers."""
    with pytest.raises(ValueError, match=r"n_bins must be positive"):
        validate_n_bins(bad)


@pytest.mark.parametrize("bad", [1.0, 2.2, "3", None, np.int64(5)])
def test_validate_n_bins_requires_python_int(bad):
    """Tests that validate_n_bins requires a Python int."""
    with pytest.raises(TypeError, match=r"n_bins must be an integer"):
        validate_n_bins(bad)


def test_validate_n_bins_allow_one_false_rejects_one():
    """Tests that validate_n_bins rejects 1 when allow_one is False."""
    with pytest.raises(ValueError, match=r"greater than 1"):
        validate_n_bins(1, allow_one=False)


def test_validate_n_bins_max_bins_enforced():
    """Tests that validate_n_bins enforces the max_bins limit."""
    with pytest.raises(ValueError, match=r"too large"):
        validate_n_bins(11, max_bins=10)


def test_validate_interval_accepts_linear_interval():
    """Tests that validate_interval accepts valid linear intervals."""
    validate_interval(0.0, 1.0, 10, log=False)


def test_validate_interval_accepts_log_interval_positive():
    """Tests that validate_interval accepts valid logarithmic intervals."""
    validate_interval(0.1, 10.0, 4, log=True)


@pytest.mark.parametrize("x_min, x_max", [(np.nan, 1.0), (0.0, np.inf), (-np.inf, 1.0)])
def test_validate_interval_rejects_non_finite(x_min, x_max):
    """Tests that validate_interval rejects non-finite bounds."""
    with pytest.raises(ValueError, match=r"must be finite"):
        validate_interval(x_min, x_max, 2, log=False)


@pytest.mark.parametrize("x_min, x_max", [(1.0, 1.0), (2.0, 1.0)])
def test_validate_interval_rejects_non_increasing_bounds(x_min, x_max):
    """Tests that validate_interval rejects non-increasing bounds."""
    with pytest.raises(ValueError, match=r"x_max must be greater than x_min"):
        validate_interval(x_min, x_max, 2, log=False)


@pytest.mark.parametrize("x_min, x_max", [(0.0, 1.0), (-1.0, 1.0)])
def test_validate_interval_log_requires_positive_bounds(x_min, x_max):
    """Tests that validate_interval rejects non-positive bounds for log=True."""
    with pytest.raises(ValueError, match=r"x_min > 0.*x_max > 0|positive"):
        validate_interval(x_min, x_max, 2, log=True)


def test_validate_interval_uses_validate_n_bins():
    """Tests that validate_interval calls validate_n_bins internally."""
    with pytest.raises(ValueError, match=r"n_bins must be positive"):
        validate_interval(0.0, 1.0, 0, log=False)


def test_validate_axis_and_weights_happy_path_returns_float_arrays():
    """Tests that validate_axis_and_weights returns float np.arrays on valid input."""
    x = [0, 1, 2, 3]
    w = [1, 2, 3, 4]
    x_out, w_out = validate_axis_and_weights(x, w)

    assert isinstance(x_out, np.ndarray)
    assert isinstance(w_out, np.ndarray)
    assert x_out.dtype == float
    assert w_out.dtype == float
    assert x_out.shape == (4,)
    assert w_out.shape == (4,)


def test_validate_axis_and_weights_requires_same_shape():
    """Tests that validate_axis_and_weights requires x and weights
    to have the same shape."""
    x = [0, 1, 2]
    w = [1, 2]
    with pytest.raises(ValueError, match=r"same shape"):
        validate_axis_and_weights(x, w)


def test_validate_axis_and_weights_requires_1d_x():
    """Tests that validate_axis_and_weights requires x to be 1D."""
    x = [[0, 1], [2, 3]]
    w = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match=r"x must be 1D"):
        validate_axis_and_weights(x, w)


def test_validate_axis_and_weights_requires_1d_weights():
    """Tests that validate_axis_and_weights requires weights to be 1D."""
    x = np.array([0, 1, 2, 3], dtype=float)
    w = np.ones((4, 1), dtype=float)
    with pytest.raises(ValueError, match=r"weights must be 1D"):
        validate_axis_and_weights(x, w)


def test_validate_axis_and_weights_rejects_non_finite_x():
    """Tests that validate_axis_and_weights rejects non-finite x values."""
    x = [0, 1, np.nan]
    w = [1, 1, 1]
    with pytest.raises(ValueError, match=r"x must contain only finite"):
        validate_axis_and_weights(x, w)


def test_validate_axis_and_weights_rejects_non_finite_weights():
    """Tests that validate_axis_and_weights rejects non-finite weight values."""
    x = [0, 1, 2]
    w = [1, np.inf, 1]
    with pytest.raises(ValueError, match=r"weights must contain only finite"):
        validate_axis_and_weights(x, w)


def test_validate_axis_and_weights_requires_at_least_two_points():
    """Tests that validate_axis_and_weights requires at least two points in x."""
    x = [0.0]
    w = [1.0]
    with pytest.raises(ValueError, match=r"at least two points"):
        validate_axis_and_weights(x, w)


@pytest.mark.parametrize(
    "x",
    [
        [0, 0, 1],  # not strictly increasing
        [0, 2, 1],  # decreasing
        [0, 1, 1.0],  # plateau
    ],
)
def test_validate_axis_and_weights_requires_strictly_increasing_x(x):
    """Tests that validate_axis_and_weights requires x to be strictly increasing."""
    w = np.ones(len(x))
    with pytest.raises(ValueError, match=r"strictly increasing"):
        validate_axis_and_weights(x, w)


def test_validate_mixed_segments_happy_path_no_total():
    """Tests that validate_mixed_segments accepts valid segments
    without total_n_bins."""
    segments = [
        {"method": "eq", "n_bins": 3},
        {"method": "log", "n_bins": 2, "params": {"base": 10}},
    ]
    validate_mixed_segments(segments)


def test_validate_mixed_segments_happy_path_with_total():
    """Tests that validate_mixed_segments accepts valid segments with total_n_bins."""
    segments = [
        {"method": "equidistant", "n_bins": 2},
        {"method": "equal_number", "n_bins": 5, "params": {}},
    ]
    validate_mixed_segments(segments, total_n_bins=7)


def test_validate_mixed_segments_rejects_empty():
    """Tests that validate_mixed_segments rejects an empty segments list."""
    with pytest.raises(ValueError, match=r"non-empty sequence"):
        validate_mixed_segments([])


def test_validate_mixed_segments_requires_mapping():
    """Tests that validate_mixed_segments requires each segment to be a mapping."""
    segments = [("eq", 2)]
    with pytest.raises(TypeError, match=r"must be a mapping"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_requires_method_and_n_bins_keys():
    """Tests that validate_mixed_segments requires 'method' and 'n_bins' keys."""
    segments = [{"method": "eq"}]
    with pytest.raises(
        ValueError, match=r"must contain at least 'method' and 'n_bins'"
    ):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_method_must_be_str():
    """Tests that validate_mixed_segments requires 'method' to be a string."""
    segments = [{"method": 123, "n_bins": 2}]
    with pytest.raises(TypeError, match=r"'method' must be a string"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_n_bins_must_be_int():
    """Tests that validate_mixed_segments requires 'n_bins' to be an int."""
    segments = [{"method": "eq", "n_bins": 2.0}]
    with pytest.raises(TypeError, match=r"'n_bins' must be an int"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_unknown_method_raises():
    """Tests that validate_mixed_segments raises for unknown binning methods."""
    segments = [{"method": "nope", "n_bins": 2}]
    with pytest.raises(ValueError, match=r"Unknown binning method"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_invalid_n_bins_raises():
    """Tests that validate_mixed_segments raises for invalid n_bins values."""
    segments = [{"method": "eq", "n_bins": 0}]
    with pytest.raises(ValueError, match=r"n_bins must be positive"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_params_must_be_mapping_if_provided():
    """Tests that validate_mixed_segments requires 'params' to be a mapping if given."""
    segments = [{"method": "eq", "n_bins": 2, "params": ["not", "a", "mapping"]}]
    with pytest.raises(TypeError, match=r"'params' must be a mapping"):
        validate_mixed_segments(segments)


def test_validate_mixed_segments_total_n_bins_mismatch_raises():
    """Tests that validate_mixed_segments raises when sum of n_bins
    does not match total_n_bins."""
    segments = [{"method": "eq", "n_bins": 2}, {"method": "log", "n_bins": 3}]
    with pytest.raises(
        ValueError,
        match=r"Sum of segment n_bins is .*total_n_bins is 6",
    ):
        validate_mixed_segments(segments, total_n_bins=6)
