"""Unit tests for binny.tomography.specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from binny.tomography.specs import (
    _builder_kwargs_from_spec,
    _load_predefined_bin_source,
    _parse_bins,
    _parse_entry,
)


def test_parse_bins_with_scheme() -> None:
    """Tests that _parse_bins accepts scheme-based bins."""
    bins = _parse_bins({"scheme": "equidistant", "n_bins": 3})
    assert bins["scheme"] == "equidistant"
    assert bins["n_bins"] == 3


def test_parse_bins_with_edges() -> None:
    """Tests that _parse_bins accepts explicit edges."""
    bins = _parse_bins({"edges": [0.0, 0.5, 1.0]})
    assert "edges" in bins
    assert bins["edges"].size == 3


def test_parse_entry_handles_optional_fields() -> None:
    """Tests that _parse_entry handles optional fields correctly."""
    entry = {
        "kind": "photoz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"scheme": "equidistant", "n_bins": 2},
    }

    spec = _parse_entry(entry)

    assert spec["kind"] == "photoz"
    assert spec["sample"] is None
    assert spec["sample_properties"] == {}
    assert spec["normalize_bins"] is True
    assert isinstance(spec["bins"], dict)


def test_parse_entry_preserves_sample_properties() -> None:
    """Tests that _parse_entry preserves sample-level observational metadata."""
    entry = {
        "role": "source",
        "kind": "photoz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"scheme": "equidistant", "n_bins": 2},
        "sample_properties": {
            "number_density": {
                "n_gal_arcmin2": 10.0,
            },
            "shape_noise": {
                "sigma_e": 0.26,
            },
            "shot_noise": {
                "model": "poisson",
            },
            "footprint": {
                "survey_area": 12000.0,
                "frac_sky": 0.29,
            },
        },
    }

    spec = _parse_entry(entry)
    props = spec["sample_properties"]

    assert props["number_density"]["n_gal_arcmin2"] == pytest.approx(10.0)
    assert props["shape_noise"]["sigma_e"] == pytest.approx(0.26)
    assert props["shot_noise"]["model"] == "poisson"
    assert props["footprint"]["survey_area"] == pytest.approx(12000.0)
    assert props["footprint"]["frac_sky"] == pytest.approx(0.29)


def test_parse_entry_rejects_non_mapping_sample_properties() -> None:
    """Tests that _parse_entry rejects malformed sample_properties."""
    with pytest.raises(ValueError, match="sample_properties must be a mapping"):
        _parse_entry(
            {
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "sample_properties": 3,
            }
        )


@pytest.mark.parametrize(
    "bad_key",
    [
        "number_density",
        "shape_noise",
        "shot_noise",
        "volume",
        "footprint",
    ],
)
def test_parse_entry_rejects_non_mapping_sample_property_blocks(
    bad_key: str,
) -> None:
    """Tests that known sample_properties blocks must be mappings."""
    with pytest.raises(
        ValueError,
        match=rf"sample_properties.{bad_key} must be a mapping",
    ):
        _parse_entry(
            {
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "sample_properties": {
                    bad_key: 1.0,
                },
            }
        )


def test_parse_entry_preserves_normalize_bins() -> None:
    """Tests that _parse_entry preserves the normalize_bins flag."""
    entry = {
        "kind": "specz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"edges": [0.0, 0.5, 1.0]},
        "normalize_bins": False,
    }

    spec = _parse_entry(entry)

    assert spec["normalize_bins"] is False


def test_parse_entry_rejects_non_bool_normalize_bins() -> None:
    """Tests that _parse_entry rejects non-boolean normalize_bins values."""
    with pytest.raises(ValueError, match="normalize_bins must be a boolean"):
        _parse_entry(
            {
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "normalize_bins": "true",
            }
        )


def test_parse_bins_rejects_edges_with_scheme() -> None:
    """Tests that _parse_bins rejects conflicting binning modes."""
    with pytest.raises(ValueError, match="must specify exactly one of"):
        _parse_bins({"edges": [0.0, 1.0], "scheme": "equidistant", "n_bins": 2})


def test_parse_bins_rejects_bad_edges_shape() -> None:
    """Tests that _parse_bins rejects non-1D or too-short edges."""
    with pytest.raises(ValueError, match="bins.edges must be a 1D sequence"):
        _parse_bins({"edges": [0.0]})


def test_parse_entry_rejects_bad_kind() -> None:
    """Tests that _parse_entry rejects unsupported kind values."""
    with pytest.raises(
        ValueError,
        match="photoz.*specz.*predefined",
    ):
        _parse_entry(
            {
                "kind": "nope",
                "nz": {
                    "model": "smail",
                    "params": {},
                },
                "bins": {
                    "scheme": "x",
                    "n_bins": 2,
                },
            }
        )


def test_parse_entry_rejects_non_mapping_uncertainties() -> None:
    """Tests that _parse_entry rejects non-mapping uncertainties."""
    with pytest.raises(ValueError, match="uncertainties must be a mapping"):
        _parse_entry(
            {
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "uncertainties": [1, 2],
            }
        )


def test_parse_entry_accepts_tabulated_source_block() -> None:
    """Tests that _parse_entry preserves tabulated source blocks."""
    entry = {
        "role": "lens",
        "sample": "lrg",
        "kind": "specz",
        "nz": {
            "model": "tabulated",
            "source": {
                "path": "desi_lrg_nz.txt",
                "z_col": 0,
                "nz_col": 1,
                "skiprows": 0,
            },
            "params": {"normalize": True},
        },
        "bins": {"edges": [0.4, 1.0]},
    }

    spec = _parse_entry(entry)

    assert spec["role"] == "lens"
    assert spec["sample"] == "lrg"
    assert spec["kind"] == "specz"
    assert spec["nz"]["model"] == "tabulated"
    assert spec["nz"]["source"]["path"] == "desi_lrg_nz.txt"
    assert np.allclose(spec["bins"]["edges"], [0.4, 1.0])


def test_parse_bins_with_scheme_and_range() -> None:
    """Tests that _parse_bins preserves scheme bin ranges."""
    bins = _parse_bins(
        {
            "scheme": "equidistant",
            "n_bins": 3,
            "range": [0.2, 1.4],
        }
    )

    assert bins["scheme"] == "equidistant"
    assert bins["n_bins"] == 3
    assert bins["range"] == (0.2, 1.4)


def test_parse_bins_rejects_missing_n_bins() -> None:
    """Tests that _parse_bins requires n_bins for scheme-based bins."""
    with pytest.raises(ValueError, match="bins.n_bins is required"):
        _parse_bins({"scheme": "equidistant"})


def test_parse_bins_rejects_missing_edges_or_scheme() -> None:
    """Tests that _parse_bins requires exactly one binning mode."""
    with pytest.raises(ValueError, match="must specify exactly one of"):
        _parse_bins({})


def test_builder_kwargs_from_spec_uses_edges() -> None:
    """Tests that explicit bin edges are passed through unchanged."""
    spec = {
        "bins": {"edges": np.array([0.0, 0.5, 1.0])},
        "uncertainties": {"sigma_z": 0.05},
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert np.allclose(kwargs["bins"]["edges"], [0.0, 0.5, 1.0])
    assert kwargs["params"]["sigma_z"] == pytest.approx(0.05)
    assert "bin_edges" not in kwargs
    assert "binning_scheme" not in kwargs
    assert "n_bins" not in kwargs


def test_builder_kwargs_from_spec_uses_scheme_and_range() -> None:
    """Tests that scheme, bin count, and range remain in the bins mapping."""
    spec = {
        "bins": {
            "scheme": "equidistant",
            "n_bins": 4,
            "range": (0.2, 1.2),
        },
        "uncertainties": {"scatter_scale": 0.03},
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert kwargs["bins"] == spec["bins"]
    assert kwargs["params"]["scatter_scale"] == pytest.approx(0.03)
    assert "bin_edges" not in kwargs
    assert "binning_scheme" not in kwargs
    assert "n_bins" not in kwargs
    assert "bin_range" not in kwargs


def test_builder_kwargs_from_spec_preserves_bin_range() -> None:
    """Tests that the configured bin range stays in the bins mapping."""
    spec = {
        "bins": {
            "scheme": "equidistant",
            "n_bins": 4,
            "range": (0.2, 1.2),
        },
        "uncertainties": {"scatter_scale": 0.03},
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert kwargs["bins"]["range"] == (0.2, 1.2)
    assert "bin_range" not in kwargs


def test_parse_bins_accepts_predefined_configuration() -> None:
    """Tests that _parse_bins preserves valid predefined-bin settings."""
    bins = _parse_bins(
        {
            "scheme": "predefined",
            "n_bins": 2,
            "source": {
                "path": "mock_bins.txt",
                "z_col": "redshift",
                "bin_cols": {
                    0: "low",
                    1: "high",
                },
            },
            "normalize_bins": True,
            "norm_method": "simpson",
        }
    )

    assert bins["scheme"] == "predefined"
    assert bins["n_bins"] == 2
    assert bins["source"]["path"] == "mock_bins.txt"
    assert bins["source"]["z_col"] == "redshift"
    assert bins["source"]["bin_cols"] == {
        0: "low",
        1: "high",
    }
    assert bins["normalize_bins"] is True
    assert bins["norm_method"] == "simpson"


@pytest.mark.parametrize(
    ("bins_block", "match"),
    [
        (
            {
                "scheme": "predefined",
            },
            "bins.source must be a mapping",
        ),
        (
            {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
                "range": [0.0, 2.0],
            },
            "must not provide bins.range",
        ),
        (
            {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
                "normalize_bins": "true",
            },
            "normalize_bins must be a boolean",
        ),
        (
            {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
                "norm_method": "rectangle",
            },
            "norm_method must be",
        ),
        (
            {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
                "n_bins": True,
            },
            "n_bins must be an integer",
        ),
        (
            {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
                "n_bins": 0,
            },
            "n_bins must be positive",
        ),
    ],
)
def test_parse_bins_rejects_invalid_predefined_configuration(
    bins_block: dict[str, Any],
    match: str,
) -> None:
    """Tests that _parse_bins rejects malformed predefined-bin settings."""
    with pytest.raises(
        ValueError,
        match=match,
    ):
        _parse_bins(bins_block)


def test_parse_entry_accepts_predefined_without_parent_nz() -> None:
    """Tests that _parse_entry accepts predefined tomography without an nz block."""
    spec = _parse_entry(
        {
            "role": "source",
            "sample": "external_sample",
            "kind": "predefined",
            "bins": {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                    "z_col": "z",
                    "bin_cols": {
                        0: "bin_0",
                        1: "bin_1",
                    },
                },
                "normalize_bins": True,
                "norm_method": "trapezoid",
            },
        }
    )

    assert spec["kind"] == "predefined"
    assert spec["nz"] is None
    assert spec["normalize_bins"] is True
    assert spec["bins"]["scheme"] == "predefined"
    assert spec["bins"]["normalize_bins"] is True


def test_parse_entry_infers_predefined_kind_from_bin_scheme() -> None:
    """Tests that _parse_entry infers predefined kind from the bin scheme."""
    spec = _parse_entry(
        {
            "role": "source",
            "bins": {
                "scheme": "predefined",
                "source": {
                    "path": "mock_bins.txt",
                },
            },
        }
    )

    assert spec["kind"] == "predefined"
    assert spec["nz"] is None
    assert spec["normalize_bins"] is False


def test_parse_entry_rejects_predefined_kind_with_generated_bins() -> None:
    """Tests that predefined entries cannot use generated binning schemes."""
    with pytest.raises(
        ValueError,
        match="kind='predefined'.*scheme='predefined'",
    ):
        _parse_entry(
            {
                "kind": "predefined",
                "bins": {
                    "scheme": "equidistant",
                    "n_bins": 2,
                },
            }
        )


@pytest.mark.parametrize(
    "kind",
    [
        "photoz",
        "specz",
    ],
)
def test_parse_entry_requires_parent_nz_for_generated_tomography(
    kind: str,
) -> None:
    """Tests that generated photo-z and spec-z entries still require nz."""
    with pytest.raises(
        ValueError,
        match="must contain an 'nz' mapping",
    ):
        _parse_entry(
            {
                "kind": kind,
                "bins": {
                    "scheme": "equidistant",
                    "n_bins": 2,
                },
            }
        )


def test_load_predefined_bin_source_reads_absolute_table(
    tmp_path: Path,
) -> None:
    """Tests that _load_predefined_bin_source reads configured named columns."""
    z = np.linspace(
        0.0,
        1.0,
        11,
        dtype=np.float64,
    )
    low = 1.0 - z
    high = z

    path = tmp_path / "predefined_bins.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                low,
                high,
            ]
        ),
        header="z bin_0 bin_1",
        comments="",
        fmt="%.18e",
    )

    loaded_z, loaded_bins = _load_predefined_bin_source(
        {
            "scheme": "predefined",
            "n_bins": 2,
            "source": {
                "path": str(path),
                "z_col": "z",
                "bin_cols": {
                    0: "bin_0",
                    1: "bin_1",
                },
            },
        }
    )

    np.testing.assert_allclose(
        loaded_z,
        z,
        rtol=0.0,
        atol=0.0,
    )
    assert list(loaded_bins) == [0, 1]

    np.testing.assert_allclose(
        loaded_bins[0],
        low,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded_bins[1],
        high,
        rtol=0.0,
        atol=0.0,
    )


def test_load_predefined_bin_source_rejects_missing_path() -> None:
    """Tests that _load_predefined_bin_source requires a source path."""
    with pytest.raises(
        ValueError,
        match="source must contain a 'path' field",
    ):
        _load_predefined_bin_source(
            {
                "scheme": "predefined",
                "source": {
                    "z_col": "z",
                },
            }
        )


def test_load_predefined_bin_source_rejects_non_mapping_bin_columns(
    tmp_path: Path,
) -> None:
    """Tests that _load_predefined_bin_source requires mapped bin columns."""
    path = tmp_path / "predefined_bins.txt"

    np.savetxt(
        path,
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        header="z bin_0",
        comments="",
    )

    with pytest.raises(
        ValueError,
        match="bin_cols must be a mapping",
    ):
        _load_predefined_bin_source(
            {
                "scheme": "predefined",
                "source": {
                    "path": str(path),
                    "z_col": "z",
                    "bin_cols": ["bin_0"],
                },
            }
        )


def test_load_predefined_bin_source_validates_n_bins(
    tmp_path: Path,
) -> None:
    """Tests that _load_predefined_bin_source validates configured bin counts."""
    z = np.linspace(
        0.0,
        1.0,
        11,
        dtype=np.float64,
    )

    path = tmp_path / "predefined_bins.txt"

    np.savetxt(
        path,
        np.column_stack(
            [
                z,
                1.0 - z,
                z,
            ]
        ),
        header="z bin_0 bin_1",
        comments="",
        fmt="%.18e",
    )

    with pytest.raises(
        ValueError,
        match="Configured n_bins=3 does not match the 2 predefined bins",
    ):
        _load_predefined_bin_source(
            {
                "scheme": "predefined",
                "n_bins": 3,
                "source": {
                    "path": str(path),
                    "z_col": "z",
                    "bin_cols": {
                        0: "bin_0",
                        1: "bin_1",
                    },
                },
            }
        )


def test_parse_bins_accepts_mixed_segments() -> None:
    """Tests that _parse_bins accepts mixed binning schemes."""
    segments = [
        {
            "scheme": "equidistant",
            "n_bins": 2,
            "x_min": 0.4,
            "x_max": 0.9,
        },
        {
            "scheme": "equal_number",
            "n_bins": 3,
            "x_min": 0.9,
            "x_max": 2.4,
        },
    ]

    bins = _parse_bins({"segments": segments})

    assert len(bins["segments"]) == 2
    assert bins["segments"][0]["scheme"] == "equidistant"
    assert bins["segments"][0]["n_bins"] == 2
    assert bins["segments"][1]["scheme"] == "equal_number"
    assert bins["segments"][1]["n_bins"] == 3
    assert bins["segments"][0]["x_max"] == bins["segments"][1]["x_min"]


def test_parse_bins_preserves_mixed_segment_params() -> None:
    """Tests that _parse_bins preserves scheme-specific segment parameters."""
    bins = _parse_bins(
        {
            "segments": [
                {
                    "scheme": "equidistant",
                    "n_bins": 2,
                    "x_min": 0.0,
                    "x_max": 1.0,
                    "params": {"custom": 1.5},
                }
            ]
        }
    )

    assert bins["segments"][0]["params"] == {"custom": 1.5}


@pytest.mark.parametrize(
    "extra",
    [
        {"scheme": "equidistant"},
        {"edges": [0.0, 1.0]},
        {"n_bins": 5},
        {"range": [0.0, 2.0]},
    ],
)
def test_parse_bins_rejects_mixed_modes(extra: dict[str, Any]) -> None:
    """Tests that mixed segments cannot be combined with other binning modes."""
    bins = {
        "segments": [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "x_min": 0.0,
                "x_max": 1.0,
            }
        ],
        **extra,
    }

    with pytest.raises(ValueError):
        _parse_bins(bins)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        "invalid",
        [{"scheme": "equidistant", "n_bins": 2}],
        [
            {
                "scheme": "equidistant",
                "n_bins": 0,
                "x_min": 0.0,
                "x_max": 1.0,
            }
        ],
        [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "x_min": 1.0,
                "x_max": 0.0,
            }
        ],
        [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "x_min": 0.0,
                "x_max": 1.0,
                "unknown": True,
            }
        ],
    ],
)
def test_parse_bins_rejects_invalid_mixed_segments(segments: Any) -> None:
    """Tests that malformed mixed binning segments raise ValueError."""
    with pytest.raises(ValueError):
        _parse_bins({"segments": segments})


@pytest.mark.parametrize(
    "second_start",
    [0.8, 1.2],
)
def test_parse_bins_rejects_nonadjacent_segments(
    second_start: float,
) -> None:
    """Tests that mixed binning segments cannot overlap or have gaps."""
    bins = {
        "segments": [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "x_min": 0.0,
                "x_max": 1.0,
            },
            {
                "scheme": "equal_number",
                "n_bins": 3,
                "x_min": second_start,
                "x_max": 2.0,
            },
        ]
    }

    with pytest.raises(ValueError, match="must be adjacent"):
        _parse_bins(bins)


def test_builder_kwargs_from_spec_preserves_mixed_segments() -> None:
    """Tests that mixed binning specifications reach the builder unchanged."""
    bins = {
        "segments": [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "x_min": 0.0,
                "x_max": 1.0,
            },
            {
                "scheme": "equal_number",
                "n_bins": 3,
                "x_min": 1.0,
                "x_max": 2.0,
            },
        ]
    }

    kwargs = _builder_kwargs_from_spec(
        {
            "bins": bins,
            "uncertainties": {"scatter_scale": 0.03},
        }
    )

    assert kwargs["bins"] == bins
    assert kwargs["params"]["scatter_scale"] == pytest.approx(0.03)


@pytest.mark.parametrize("normalize_bins", [False, True])
def test_builder_kwargs_from_spec_forwards_normalize_bins(normalize_bins):
    """Tests that the configured bin normalization flag is forwarded."""
    spec = {
        "bins": {"edges": [0.0, 0.5, 1.0]},
        "normalize_bins": normalize_bins,
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert kwargs["normalize_bins"] is normalize_bins


def test_builder_kwargs_from_spec_normalization_default():
    """Tests that bin normalization is enabled by default."""
    spec = {
        "bins": {"edges": [0.0, 0.5, 1.0]},
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert kwargs["normalize_bins"] is True


def test_builder_kwargs_from_spec_preserves_normalization_override():
    """Tests that explicit builder normalization settings are preserved."""
    spec = {
        "bins": {"edges": [0.0, 0.5, 1.0]},
        "normalize_bins": True,
        "uncertainties": {
            "normalize_input": False,
            "normalize_bins": False,
            "norm_method": "simpson",
        },
    }

    kwargs = _builder_kwargs_from_spec(spec)

    assert kwargs["normalize_input"] is False
    assert kwargs["normalize_bins"] is False
    assert kwargs["norm_method"] == "simpson"
