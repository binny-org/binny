import numpy as np
import pytest

from binny.utils.broadcasting import as_per_bin


def test_as_per_bin_scalar_int_broadcasts_float_array() -> None:
    """Test that an integer scalar is broadcast to a float array."""
    out = as_per_bin(3, n_bins=4, name="sigma")
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert out.dtype == float
    np.testing.assert_allclose(out, [3.0, 3.0, 3.0, 3.0])


def test_as_per_bin_scalar_float_broadcasts_float_array() -> None:
    """Test that a float scalar is broadcast to a float array."""
    out = as_per_bin(1.25, n_bins=3, name="sigma")
    assert out.dtype == float
    np.testing.assert_allclose(out, [1.25, 1.25, 1.25])


def test_as_per_bin_none_returns_object_array_of_none() -> None:
    """Test that None is broadcast to an object array of None."""
    out = as_per_bin(None, n_bins=5, name="sigma")
    assert out.shape == (5,)
    assert out.dtype == object
    assert out.tolist() == [None, None, None, None, None]


def test_as_per_bin_sequence_correct_length_all_numeric_returns_float_array() -> None:
    """Test that a numeric sequence of correct length returns a float array."""
    out = as_per_bin([1, 2.5, "3"], n_bins=3, name="sigma")
    assert out.dtype == float
    np.testing.assert_allclose(out, [1.0, 2.5, 3.0])


def test_as_per_bin_sequence_correct_length_with_none_returns_object_array() -> None:
    """Test that a sequence with None returns an object array."""
    out = as_per_bin([1, None, 3.5], n_bins=3, name="sigma")
    assert out.dtype == object
    assert out.tolist() == [1.0, None, 3.5]


def test_as_per_bin_tuple_sequence_supported() -> None:
    """Test that a tuple sequence is supported."""
    out = as_per_bin((0, 1), n_bins=2, name="sigma")
    assert out.dtype == float
    np.testing.assert_allclose(out, [0.0, 1.0])


def test_as_per_bin_sequence_length_mismatch_raises_value_error() -> None:
    """Test that a sequence with incorrect length raises ValueError."""
    with pytest.raises(ValueError, match=r"sigma must have length 4, got 3\."):
        as_per_bin([1, 2, 3], n_bins=4, name="sigma")


def test_as_per_bin_n_bins_less_than_one_raises_value_error() -> None:
    """Test that n_bins < 1 raises ValueError."""
    with pytest.raises(ValueError, match=r"n_bins must be >= 1\."):
        as_per_bin(1.0, n_bins=0, name="sigma")


def test_as_per_bin_invalid_type_raises_type_error_with_name() -> None:
    """Test that an invalid type raises TypeError with the parameter name."""
    # object() is not iterable -> triggers TypeError in list(x) path
    with pytest.raises(TypeError, match=r"sigma must be scalar, None, or a sequence\."):
        as_per_bin(object(), n_bins=2, name="sigma")


def test_as_per_bin_string_length_mismatch() -> None:
    """Test that a string sequence with incorrect length raises ValueError."""
    with pytest.raises(ValueError, match=r"sigma must have length 2, got 3\."):
        as_per_bin("abc", n_bins=2, name="sigma")
