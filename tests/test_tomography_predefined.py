"""Tests for predefined tomography and file loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from binny import (
    Tomography,
    build_predefined_tomography,
    load_predefined_bins,
)


@pytest.fixture
def synthetic_bins() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return synthetic, unnormalized tomographic distributions."""
    z = np.linspace(0.0, 2.0, 201, dtype=np.float64)
    bins = {
        0: 1.4 * np.exp(-0.5 * ((z - 0.45) / 0.18) ** 2),
        1: 0.9 * np.exp(-0.5 * ((z - 1.05) / 0.24) ** 2),
        2: 0.6 * np.exp(-0.5 * ((z - 1.60) / 0.20) ** 2),
    }
    return z, bins


def _write_named_table(
    path: Path,
    z: np.ndarray,
    bins: dict[int, np.ndarray],
) -> None:
    """Write predefined distributions with named columns."""
    keys = sorted(bins)
    table = np.column_stack([z, *(bins[key] for key in keys)])
    header = " ".join(["z", *(f"bin_{key}" for key in keys)])
    np.savetxt(path, table, header=header, comments="", fmt="%.18e")


def test_build_predefined_tomography(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that tomography is constructed directly from supplied bin curves."""
    z, bins = synthetic_bins

    result = build_predefined_tomography(
        z,
        {2: bins[2], 0: bins[0], 1: bins[1]},
    )

    assert isinstance(result, Tomography)
    assert result.n_bins == 3
    assert result.bin_indices == [0, 1, 2]
    assert result.metadata == {}

    np.testing.assert_array_equal(result.z, z)
    np.testing.assert_allclose(
        result.nz,
        np.sum(np.stack(list(bins.values())), axis=0),
    )

    for key, curve in bins.items():
        np.testing.assert_array_equal(result.bins[key], curve)
        np.testing.assert_array_equal(result[key], curve)
        assert not np.shares_memory(result.bins[key], curve)


def test_build_predefined_tomography_metadata(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that population metadata uses raw bin integrals."""
    z, bins = synthetic_bins

    result = build_predefined_tomography(
        z,
        bins,
        include_metadata=True,
    )

    metadata = result.metadata
    assert metadata
    assert metadata["kind"] == "predefined"
    assert metadata["inputs"]["parent_source"] == "sum_of_bins"
    assert metadata["inputs"]["normalize_bins"] is False
    assert metadata["inputs"]["norm_method"] == "trapezoid"

    expected_norms = {key: float(np.trapezoid(curve, x=z)) for key, curve in bins.items()}
    assert metadata["bins_norms"] == pytest.approx(expected_norms)
    assert metadata["parent_norm"] == pytest.approx(np.trapezoid(result.nz, x=z))
    assert metadata["frac_per_bin"] == pytest.approx(
        {key: area / metadata["parent_norm"] for key, area in expected_norms.items()}
    )
    assert sum(metadata["frac_per_bin"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["trapezoid", "simpson"])
def test_build_predefined_tomography_normalization(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
    method: str,
) -> None:
    """Tests that output bins are normalized without modifying the raw parent."""
    z, bins = synthetic_bins

    result = build_predefined_tomography(
        z,
        bins,
        normalize=True,
        norm_method=method,
        include_metadata=True,
    )

    raw_parent = np.sum(np.stack(list(bins.values())), axis=0)
    np.testing.assert_allclose(result.nz, raw_parent)

    for curve in result.bins.values():
        if method == "trapezoid":
            area = np.trapezoid(curve, x=z)
        else:
            from scipy.integrate import simpson

            area = simpson(curve, x=z)
        assert area == pytest.approx(1.0)

    assert result.metadata
    assert result.metadata["inputs"]["normalize_bins"] is True
    assert result.metadata["inputs"]["norm_method"] == method


def test_build_predefined_tomography_provided_parent(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that an explicitly supplied parent distribution is preserved."""
    z, bins = synthetic_bins
    parent = 2.5 * np.sum(np.stack(list(bins.values())), axis=0)

    result = build_predefined_tomography(
        z,
        bins,
        nz=parent,
        include_metadata=True,
    )

    np.testing.assert_array_equal(result.nz, parent)
    assert not np.shares_memory(result.nz, parent)

    assert result.metadata
    assert result.metadata["inputs"]["parent_source"] == "provided"

    parent_norm = np.trapezoid(parent, x=z)
    for key, curve in bins.items():
        expected = np.trapezoid(curve, x=z) / parent_norm
        assert result.metadata["frac_per_bin"][key] == pytest.approx(expected)


def test_build_predefined_tomography_nonconsecutive_keys(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that arbitrary integer bin indices are preserved in sorted order."""
    z, bins = synthetic_bins

    result = build_predefined_tomography(
        z,
        {9: bins[2], 2: bins[0]},
    )

    assert list(result.bins) == [2, 9]
    assert result.n_bins == 2
    np.testing.assert_array_equal(result[2], bins[0])
    np.testing.assert_array_equal(result[9], bins[2])


def test_build_predefined_tomography_does_not_modify_inputs(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that input arrays remain unchanged."""
    z, bins = synthetic_bins
    original_z = z.copy()
    original_bins = {key: curve.copy() for key, curve in bins.items()}

    build_predefined_tomography(z, bins, normalize=True)

    np.testing.assert_array_equal(z, original_z)
    for key in bins:
        np.testing.assert_array_equal(bins[key], original_bins[key])


@pytest.mark.parametrize(
    ("z", "message"),
    [
        (np.array([[0.0, 1.0]]), "one-dimensional"),
        (np.array([0.0]), "at least two"),
        (np.array([0.0, np.nan, 1.0]), "finite"),
        (np.array([0.0, 0.5, 0.5]), "strictly increasing"),
        (np.array([0.0, 1.0, 0.5]), "strictly increasing"),
    ],
)
def test_build_predefined_tomography_invalid_z(
    z: np.ndarray,
    message: str,
) -> None:
    """Tests that invalid redshift grids are rejected."""
    with pytest.raises(ValueError, match=message):
        build_predefined_tomography(z, {0: np.ones(z.shape)})


@pytest.mark.parametrize("bins", [{}, [], None])
def test_build_predefined_tomography_invalid_bins(bins: object) -> None:
    """Tests that missing or invalid bin mappings are rejected."""
    z = np.linspace(0.0, 1.0, 11)

    with pytest.raises(ValueError, match="non-empty mapping"):
        build_predefined_tomography(z, bins)


@pytest.mark.parametrize("key", ["0", 0.5, True, None])
def test_build_predefined_tomography_invalid_keys(key: object) -> None:
    """Tests that noninteger bin indices are rejected."""
    z = np.linspace(0.0, 1.0, 11)

    with pytest.raises(TypeError, match="integers"):
        build_predefined_tomography(z, {key: np.ones_like(z)})


def test_build_predefined_tomography_zero_integral() -> None:
    """Tests that bins with zero integrated population are rejected."""
    z = np.linspace(0.0, 1.0, 11)

    with pytest.raises(ValueError, match="zero integral"):
        build_predefined_tomography(z, {0: np.zeros_like(z)})


def test_build_predefined_tomography_invalid_curve_shape() -> None:
    """Tests that bin curves must match the redshift grid."""
    z = np.linspace(0.0, 1.0, 11)

    with pytest.raises(ValueError):
        build_predefined_tomography(z, {0: np.ones(10)})


def test_build_predefined_tomography_invalid_parent() -> None:
    """Tests that parent distributions with invalid dimensions are rejected."""
    z = np.linspace(0.0, 1.0, 11)

    with pytest.raises(ValueError):
        build_predefined_tomography(
            z,
            {0: np.ones_like(z)},
            nz=np.ones(10),
        )


def test_load_predefined_bins_named_columns(tmp_path: Path) -> None:
    """Tests that named columns are loaded in sorted bin order."""
    z = np.linspace(0.0, 1.0, 11)
    low = z**2
    high = np.sqrt(z)

    path = tmp_path / "named.txt"
    np.savetxt(
        path,
        np.column_stack([z, low, high]),
        header="redshift low_sample high_sample",
        comments="",
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(
        path,
        z_col="redshift",
        bin_cols={5: "high_sample", 2: "low_sample"},
    )

    np.testing.assert_array_equal(loaded_z, z)
    assert list(loaded_bins) == [2, 5]
    np.testing.assert_array_equal(loaded_bins[2], low)
    np.testing.assert_array_equal(loaded_bins[5], high)


def test_load_predefined_bins_auto_detect(tmp_path: Path) -> None:
    """Tests that standard bin column names are detected automatically."""
    z = np.linspace(0.0, 1.0, 21)
    first = np.exp(-0.5 * ((z - 0.3) / 0.12) ** 2)
    second = np.exp(-0.5 * ((z - 0.7) / 0.15) ** 2)

    path = tmp_path / "auto.txt"
    np.savetxt(
        path,
        np.column_stack([z, second, np.ones_like(z), first]),
        header="z bin_3 auxiliary bin_1",
        comments="",
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(path)

    np.testing.assert_array_equal(loaded_z, z)
    assert list(loaded_bins) == [1, 3]
    np.testing.assert_allclose(loaded_bins[1], first)
    np.testing.assert_allclose(loaded_bins[3], second)


def test_load_predefined_bins_positional(tmp_path: Path) -> None:
    """Tests that headerless tables support integer column positions."""
    z = np.linspace(0.0, 1.5, 31)
    first = 1.0 + z
    second = 2.0 - 0.5 * z

    path = tmp_path / "positional.txt"
    np.savetxt(
        path,
        np.column_stack([z, first, second]),
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(
        path,
        z_col=0,
        bin_cols={4: 2, 1: 1},
    )

    np.testing.assert_array_equal(loaded_z, z)
    assert list(loaded_bins) == [1, 4]
    np.testing.assert_array_equal(loaded_bins[1], first)
    np.testing.assert_array_equal(loaded_bins[4], second)


def test_load_and_build_predefined_tomography(
    tmp_path: Path,
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that tomography can be constructed from a text file."""
    z, bins = synthetic_bins
    path = tmp_path / "predefined.txt"
    _write_named_table(path, z, bins)

    loaded_z, loaded_bins = load_predefined_bins(path)
    result = build_predefined_tomography(
        loaded_z,
        loaded_bins,
        include_metadata=True,
    )

    assert isinstance(result, Tomography)
    assert result.n_bins == 3
    np.testing.assert_allclose(result.z, z)

    for key in bins:
        np.testing.assert_allclose(result.bins[key], bins[key])

    np.testing.assert_allclose(
        result.nz,
        np.sum(np.stack(list(bins.values())), axis=0),
    )
    assert result.metadata
    assert result.metadata["kind"] == "predefined"


def test_load_predefined_bins_missing_file(tmp_path: Path) -> None:
    """Tests that nonexistent files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_predefined_bins(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    ("z_col", "bin_cols", "error", "message"),
    [
        ("z", {0: 1}, ValueError, "same type"),
        (0, {0: "bin_0"}, ValueError, "same type"),
        (0, None, ValueError, "required"),
        ("z", {}, ValueError, "non-empty mapping"),
        (True, {0: 1}, TypeError, "z_col"),
        ("z", {True: "bin_0"}, TypeError, "indices"),
        ("z", {0: True}, TypeError, "selectors"),
    ],
)
def test_load_predefined_bins_invalid_selectors(
    tmp_path: Path,
    z_col: object,
    bin_cols: object,
    error: type[Exception],
    message: str,
) -> None:
    """Tests that invalid or mixed column selectors are rejected."""
    path = tmp_path / "selectors.txt"
    z = np.linspace(0.0, 1.0, 11)
    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(error, match=message):
        load_predefined_bins(path, z_col=z_col, bin_cols=bin_cols)


def test_load_predefined_bins_unsorted_z(tmp_path: Path) -> None:
    """Tests that nonmonotonic redshift grids are rejected."""
    path = tmp_path / "unsorted.txt"
    z = np.array([0.0, 0.3, 0.2, 0.5])

    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_predefined_bins(path)


def test_load_predefined_bins_missing_column(tmp_path: Path) -> None:
    """Tests that requested columns must exist in the input file."""
    path = tmp_path / "missing_column.txt"
    z = np.linspace(0.0, 1.0, 11)

    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(ValueError, match="Missing column"):
        load_predefined_bins(path, bin_cols={0: "bin_99"})


def test_load_predefined_bins_no_bins(tmp_path: Path) -> None:
    """Tests that named tables without bin columns are rejected."""
    path = tmp_path / "no_bins.txt"
    z = np.linspace(0.0, 1.0, 11)

    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
        header="z auxiliary",
        comments="",
    )

    with pytest.raises(ValueError, match="No predefined bin columns"):
        load_predefined_bins(path)


def test_load_predefined_bins_invalid_standard_name(tmp_path: Path) -> None:
    """Tests that malformed automatic bin column names are rejected."""
    path = tmp_path / "bad_name.txt"
    z = np.linspace(0.0, 1.0, 11)

    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
        header="z bin_wrong",
        comments="",
    )

    with pytest.raises(ValueError, match="Invalid predefined column"):
        load_predefined_bins(path)


def test_load_predefined_bins_invalid_positional_index(tmp_path: Path) -> None:
    """Tests that column positions outside the input table are rejected."""
    path = tmp_path / "bad_index.txt"
    z = np.linspace(0.0, 1.0, 11)

    np.savetxt(
        path,
        np.column_stack([z, np.ones_like(z)]),
    )

    with pytest.raises(ValueError, match="outside the available range"):
        load_predefined_bins(path, z_col=0, bin_cols={0: 5})


def test_predefined_matches_legacy(
    synthetic_bins: tuple[np.ndarray, dict[int, np.ndarray]],
) -> None:
    """Tests that functional predefined tomography matches legacy calculations."""
    from binny.nz_tomo.predefined import prepare_predefined_bins

    z, bins = synthetic_bins

    for normalize in (False, True):
        for include_metadata in (False, True):
            old_z, old_parent, old_bins, old_metadata = prepare_predefined_bins(
                z,
                bins,
                normalize_bins=normalize,
                include_metadata=include_metadata,
            )
            new = build_predefined_tomography(
                z,
                bins,
                normalize=normalize,
                include_metadata=include_metadata,
            )

            np.testing.assert_allclose(new.z, old_z)
            np.testing.assert_allclose(new.nz, old_parent)

            for key in bins:
                np.testing.assert_allclose(new.bins[key], old_bins[key])

            assert new.metadata == (old_metadata if old_metadata is not None else {})
