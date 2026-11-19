"""Regression tests for metadata containing NumPy arrays."""

import numpy as np
import pytest

from binny.utils.metadata import round_floats, save_metadata_txt


def test_round_floats_nested_numpy_arrays():
    """Tests that nested NumPy arrays are rounded without losing values."""
    values = np.array([[0.12345, 0.67891], [1.23456, 2.34567]])

    result = round_floats({"values": values}, 3)

    assert result == {"values": [[0.123, 0.679], [1.235, 2.346]]}

    np.testing.assert_array_equal(
        values,
        [[0.12345, 0.67891], [1.23456, 2.34567]],
    )


@pytest.mark.parametrize("decimal_places", [None, 2, 6])
def test_save_metadata_preserves_all_array_elements(tmp_path, decimal_places):
    """Tests that large NumPy arrays are saved without truncation."""
    values = np.arange(2001, dtype=float)
    path = tmp_path / "metadata.txt"

    save_metadata_txt(
        {"values": values},
        path,
        decimal_places=decimal_places,
    )

    text = path.read_text()

    assert "..." not in text

    saved = [float(line.strip()[2:]) for line in text.splitlines() if line.strip().startswith("- ")]

    np.testing.assert_array_equal(saved, values)


def test_save_metadata_preserves_nested_array_values(tmp_path):
    """Tests that multidimensional arrays retain every element."""
    values = np.array([[0.123456789, 1.23456789], [2.5, 3.5]])
    path = tmp_path / "metadata.txt"

    save_metadata_txt(
        {"nested": {"values": values}},
        path,
        decimal_places=None,
    )

    text = path.read_text()

    assert "..." not in text

    saved = [float(line.strip()[2:]) for line in text.splitlines() if line.strip().startswith("- ")]

    np.testing.assert_array_equal(saved, values.ravel())
