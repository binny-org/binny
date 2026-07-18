"""Unit tests for ``binny.nz_tomo.predefined``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from binny import NZTomography
from binny.nz_tomo.predefined import (
    load_predefined_bins,
    prepare_predefined_bins,
)


def _synthetic_bins() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    z = np.linspace(
        0.0,
        2.0,
        201,
        dtype=np.float64,
    )

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
    table = np.column_stack(
        [
            z,
            *[bins[key] for key in sorted(bins)],
        ]
    )

    header = " ".join(
        [
            "z",
            *[f"bin_{key}" for key in sorted(bins)],
        ]
    )

    np.savetxt(
        path,
        table,
        header=header,
        comments="",
        fmt="%.18e",
    )


def _write_predefined_config(
    path: Path,
    *,
    data_path: Path,
    n_bins: int | None = None,
    normalize_bins: bool = False,
) -> None:
    bins_spec: dict[str, object] = {
        "scheme": "predefined",
        "source": {
            "path": str(data_path.resolve()),
            "z_col": "z",
            "bin_cols": {
                0: "bin_0",
                1: "bin_1",
                2: "bin_2",
            },
        },
        "normalize_bins": normalize_bins,
        "norm_method": "trapezoid",
    }

    if n_bins is not None:
        bins_spec["n_bins"] = n_bins

    config = {
        "name": "mock_predefined_test",
        "tomography": [
            {
                "role": "source",
                "sample": "mock_predefined",
                "name": "mock predefined source bins",
                "kind": "predefined",
                "bins": bins_spec,
            }
        ],
    }

    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_load_predefined_bins_with_named_columns(
    tmp_path: Path,
) -> None:
    """Tests that explicitly named columns are loaded into sorted bin keys."""
    z = np.linspace(
        0.0,
        1.0,
        11,
        dtype=np.float64,
    )
    low = z**2
    high = np.sqrt(z)

    path = tmp_path / "named_bins.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                low,
                high,
            ]
        ),
        header="redshift low_sample high_sample",
        comments="",
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(
        path,
        z_col="redshift",
        bin_cols={
            5: "high_sample",
            2: "low_sample",
        },
    )

    np.testing.assert_allclose(
        loaded_z,
        z,
        rtol=0.0,
        atol=0.0,
    )
    assert list(loaded_bins) == [2, 5]

    np.testing.assert_allclose(
        loaded_bins[2],
        low,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded_bins[5],
        high,
        rtol=0.0,
        atol=0.0,
    )


def test_load_predefined_bins_detects_standard_names(
    tmp_path: Path,
) -> None:
    """Tests that bin columns are detected automatically from standard names."""
    z = np.linspace(
        0.0,
        1.0,
        21,
        dtype=np.float64,
    )
    bin_3 = np.exp(-0.5 * ((z - 0.7) / 0.15) ** 2)
    bin_1 = np.exp(-0.5 * ((z - 0.3) / 0.12) ** 2)
    auxiliary = np.ones_like(z)

    path = tmp_path / "automatic_bins.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                bin_3,
                auxiliary,
                bin_1,
            ]
        ),
        header="z bin_3 auxiliary bin_1",
        comments="",
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(path)

    np.testing.assert_allclose(
        loaded_z,
        z,
        rtol=0.0,
        atol=0.0,
    )
    assert list(loaded_bins) == [1, 3]

    np.testing.assert_allclose(
        loaded_bins[1],
        bin_1,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded_bins[3],
        bin_3,
        rtol=0.0,
        atol=0.0,
    )


def test_load_predefined_bins_with_positional_columns(
    tmp_path: Path,
) -> None:
    """Tests that integer column positions load headerless tables correctly."""
    z = np.linspace(
        0.0,
        1.5,
        31,
        dtype=np.float64,
    )
    first = 1.0 + z
    second = 2.0 - 0.5 * z

    path = tmp_path / "positional_bins.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                first,
                second,
            ]
        ),
        fmt="%.18e",
    )

    loaded_z, loaded_bins = load_predefined_bins(
        path,
        z_col=0,
        bin_cols={
            4: 2,
            1: 1,
        },
    )

    np.testing.assert_allclose(
        loaded_z,
        z,
        rtol=0.0,
        atol=0.0,
    )
    assert list(loaded_bins) == [1, 4]

    np.testing.assert_allclose(
        loaded_bins[1],
        first,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded_bins[4],
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_load_predefined_bins_rejects_mixed_selectors(
    tmp_path: Path,
) -> None:
    """Tests that named and positional column selectors cannot be mixed."""
    z = np.linspace(
        0.0,
        1.0,
        11,
        dtype=np.float64,
    )

    path = tmp_path / "mixed_selectors.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                np.ones_like(z),
            ]
        ),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(
        ValueError,
        match="same selector type",
    ):
        load_predefined_bins(
            path,
            z_col="z",
            bin_cols={
                0: 1,
            },
        )


def test_load_predefined_bins_rejects_unsorted_redshifts(
    tmp_path: Path,
) -> None:
    """Tests that loaded redshift grids must be strictly increasing."""
    z = np.asarray(
        [
            0.0,
            0.3,
            0.2,
            0.5,
        ],
        dtype=np.float64,
    )
    curve = np.ones_like(z)

    path = tmp_path / "unsorted_redshifts.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                curve,
            ]
        ),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        load_predefined_bins(path)


def test_prepare_predefined_bins_constructs_parent_and_metadata() -> None:
    """Tests that raw bins produce a summed parent and population metadata."""
    z, bins = _synthetic_bins()

    (
        returned_z,
        parent,
        prepared,
        metadata,
    ) = prepare_predefined_bins(
        z=z,
        bins={
            2: bins[2],
            0: bins[0],
            1: bins[1],
        },
        include_metadata=True,
    )

    expected_parent = np.sum(
        np.stack(
            [
                bins[0],
                bins[1],
                bins[2],
            ],
            axis=0,
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        returned_z,
        z,
    )
    np.testing.assert_allclose(
        parent,
        expected_parent,
    )

    assert list(prepared) == [0, 1, 2]

    for key in prepared:
        np.testing.assert_allclose(
            prepared[key],
            bins[key],
        )
        assert not np.shares_memory(
            prepared[key],
            bins[key],
        )

    assert metadata is not None
    assert metadata["kind"] == "predefined"
    assert metadata["inputs"]["parent_source"] == "sum_of_bins"
    assert metadata["inputs"]["normalize_bins"] is False

    expected_norms = {
        key: float(
            np.trapezoid(
                bins[key],
                x=z,
            )
        )
        for key in bins
    }

    assert metadata["bins_norms"] == pytest.approx(expected_norms)

    assert sum(metadata["frac_per_bin"].values()) == pytest.approx(1.0)


def test_prepare_predefined_bins_normalizes_only_output_bins() -> None:
    """Tests that normalization changes returned bins but not the summed parent."""
    z, bins = _synthetic_bins()

    raw_parent = np.sum(
        np.stack(
            list(bins.values()),
            axis=0,
        ),
        axis=0,
    )

    (
        _,
        parent,
        prepared,
        metadata,
    ) = prepare_predefined_bins(
        z=z,
        bins=bins,
        normalize_bins=True,
        include_metadata=True,
    )

    np.testing.assert_allclose(
        parent,
        raw_parent,
    )

    for curve in prepared.values():
        assert np.trapezoid(
            curve,
            x=z,
        ) == pytest.approx(1.0)

    assert metadata is not None
    assert metadata["inputs"]["normalize_bins"] is True
    assert metadata["parent_norm"] == pytest.approx(
        np.trapezoid(
            raw_parent,
            x=z,
        )
    )


def test_prepare_predefined_bins_preserves_provided_parent() -> None:
    """Tests that an explicitly provided parent replaces the sum-of-bins parent."""
    z, bins = _synthetic_bins()

    provided_parent = 2.5 * np.sum(
        np.stack(
            list(bins.values()),
            axis=0,
        ),
        axis=0,
    )

    (
        _,
        parent,
        _,
        metadata,
    ) = prepare_predefined_bins(
        z=z,
        bins=bins,
        nz=provided_parent,
        include_metadata=True,
    )

    np.testing.assert_allclose(
        parent,
        provided_parent,
    )

    assert metadata is not None
    assert metadata["inputs"]["parent_source"] == "provided"

    parent_norm = np.trapezoid(
        provided_parent,
        x=z,
    )

    for key, curve in bins.items():
        expected_fraction = (
            np.trapezoid(
                curve,
                x=z,
            )
            / parent_norm
        )

        assert metadata["frac_per_bin"][key] == pytest.approx(expected_fraction)


def test_prepare_predefined_bins_rejects_zero_integral_bins() -> None:
    """Tests that bins with zero integrated population are rejected."""
    z = np.linspace(
        0.0,
        1.0,
        11,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="zero integral",
    ):
        prepare_predefined_bins(
            z=z,
            bins={
                0: np.zeros_like(z),
            },
        )


def test_build_bins_loads_predefined_config_end_to_end(
    tmp_path: Path,
) -> None:
    """Tests that config-driven predefined tomography preserves supplied curves."""
    z, bins = _synthetic_bins()

    data_path = tmp_path / "predefined_bins.txt"
    config_path = tmp_path / "predefined_config.yaml"

    _write_named_table(
        data_path,
        z,
        bins,
    )
    _write_predefined_config(
        config_path,
        data_path=data_path,
    )

    result = NZTomography().build_bins(
        config_file=config_path,
        role="source",
        sample="mock_predefined",
        include_tomo_metadata=True,
    )

    assert result.spec["kind"] == "predefined"
    assert result.spec["bins"]["scheme"] == "predefined"
    assert result.bin_keys == [0, 1, 2]

    np.testing.assert_allclose(
        result.z,
        z,
    )

    for key in result.bin_keys:
        np.testing.assert_allclose(
            result.bins[key],
            bins[key],
        )

    expected_parent = np.sum(
        np.stack(
            [
                bins[0],
                bins[1],
                bins[2],
            ],
            axis=0,
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        result.nz,
        expected_parent,
    )

    assert result.tomo_meta is not None
    assert result.tomo_meta["kind"] == "predefined"
    assert result.tomo_meta["inputs"]["parent_source"] == "sum_of_bins"

    shape_statistics = result.shape_stats()
    population_statistics = result.population_stats()

    assert set(shape_statistics["per_bin"]) == {
        0,
        1,
        2,
    }
    assert set(population_statistics["fractions"]) == {
        0,
        1,
        2,
    }
    assert sum(population_statistics["fractions"].values()) == pytest.approx(
        1.0,
        abs=0.02,
    )


def test_build_bins_validates_optional_n_bins(
    tmp_path: Path,
) -> None:
    """Tests that configured n_bins must match the loaded predefined columns."""
    z, bins = _synthetic_bins()

    data_path = tmp_path / "predefined_bins.txt"
    config_path = tmp_path / "predefined_config.yaml"

    _write_named_table(
        data_path,
        z,
        bins,
    )
    _write_predefined_config(
        config_path,
        data_path=data_path,
        n_bins=4,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        NZTomography().build_bins(
            config_file=config_path,
            role="source",
            sample="mock_predefined",
        )
