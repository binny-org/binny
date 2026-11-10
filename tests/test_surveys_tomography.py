"""Tests for building tomography from survey configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from binny import build_survey_tomography
from binny.surveys.survey_presets import load_survey_config
from binny.tomography.core import Tomography


@pytest.fixture
def photoz_config() -> dict[str, Any]:
    """Return a minimal in-memory photo-z survey configuration."""
    return {
        "name": "test_photoz",
        "z_grid": {"start": 0.0, "stop": 3.0, "n": 301},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {
                        "z0": 0.5,
                        "alpha": 2.0,
                        "beta": 1.0,
                    },
                },
                "bins": {
                    "scheme": "equidistant",
                    "n_bins": 4,
                },
                "uncertainties": {
                    "scatter_scale": 0.05,
                    "mean_offset": 0.01,
                },
                "normalize_bins": True,
            }
        ],
    }


@pytest.fixture
def specz_config() -> dict[str, Any]:
    """Return a minimal in-memory spec-z survey configuration."""
    return {
        "name": "test_specz",
        "z_grid": {"start": 0.0, "stop": 2.0, "n": 301},
        "tomography": [
            {
                "role": "lens",
                "year": "4",
                "kind": "specz",
                "nz": {
                    "model": "smail",
                    "params": {
                        "z0": 0.4,
                        "alpha": 2.0,
                        "beta": 1.0,
                    },
                },
                "bins": {
                    "scheme": "equidistant",
                    "n_bins": 4,
                },
                "uncertainties": {
                    "completeness": 1.0,
                },
                "normalize_bins": True,
            }
        ],
    }


def test_public_api() -> None:
    """Tests that the survey builder is available from the public API."""
    assert callable(build_survey_tomography)


def test_builtin_lsst_preset() -> None:
    """Tests that the LSST lens preset constructs tomography."""
    result = build_survey_tomography(
        "lsst",
        role="lens",
        year=1,
    )

    assert isinstance(result, Tomography)
    assert result.n_bins == 5
    assert result.bin_indices == [0, 1, 2, 3, 4]
    assert result.spec is not None
    assert result.spec["kind"] == "photoz"
    assert result.z.shape == result.nz.shape

    for values in result.bins.values():
        assert values.shape == result.z.shape
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    ("filename", "kind", "n_bins"),
    [
        ("example_minimal_photoz.yaml", "photoz", 4),
        ("example_minimal_specz.yaml", "specz", 4),
        ("example_full_photoz.yaml", "photoz", 4),
        ("example_full_specz.yaml", "specz", 4),
        ("mock_predefined_config.yaml", "predefined", 5),
    ],
)
def test_shipped_configurations(
    filename: str,
    kind: str,
    n_bins: int,
) -> None:
    """Tests that shipped configurations construct valid tomography."""
    result = build_survey_tomography(config_file=filename)

    assert isinstance(result, Tomography)
    assert result.spec is not None
    assert result.spec["kind"] == kind
    assert result.n_bins == n_bins
    assert result.z.shape == result.nz.shape

    for values in result.bins.values():
        assert values.shape == result.z.shape
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize("kind", ["photoz", "specz"])
def test_in_memory_configuration(
    kind: str,
    photoz_config: dict[str, Any],
    specz_config: dict[str, Any],
) -> None:
    """Tests that in-memory configurations construct tomography."""
    cfg = photoz_config if kind == "photoz" else specz_config

    result = build_survey_tomography(cfg=cfg)

    assert isinstance(result, Tomography)
    assert result.spec is not None
    assert result.spec["kind"] == kind
    assert result.n_bins == 4


def test_in_memory_matches_builtin() -> None:
    """Tests that in-memory LSST configurations reproduce built-in results."""
    cfg = load_survey_config("lsst")

    expected = build_survey_tomography("lsst", role="lens", year=1)
    actual = build_survey_tomography(cfg=cfg, role="lens", year=1)

    np.testing.assert_allclose(actual.z, expected.z)
    np.testing.assert_allclose(actual.nz, expected.nz)

    assert actual.bins.keys() == expected.bins.keys()

    for index in expected.bins:
        np.testing.assert_allclose(
            actual.bins[index],
            expected.bins[index],
            rtol=1e-12,
            atol=1e-14,
        )


def test_custom_yaml_file(
    tmp_path: Path,
    photoz_config: dict[str, Any],
) -> None:
    """Tests that a custom YAML configuration constructs tomography."""
    path = tmp_path / "custom_survey.yaml"
    path.write_text(
        yaml.safe_dump(photoz_config),
        encoding="utf-8",
    )

    result = build_survey_tomography(config_file=path)

    assert isinstance(result, Tomography)
    assert result.n_bins == 4
    assert result.z.size == 301
    assert result.spec is not None
    assert result.spec["kind"] == "photoz"


def test_redshift_grid_override(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that an explicit redshift grid replaces the configured grid."""
    z = np.linspace(0.0, 3.0, 151)

    result = build_survey_tomography(
        cfg=photoz_config,
        z=z,
    )

    np.testing.assert_array_equal(result.z, z)
    assert result.nz.shape == z.shape

    for values in result.bins.values():
        assert values.shape == z.shape


def test_overrides(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that overrides modify the resolved tomography specification."""
    result = build_survey_tomography(
        cfg=photoz_config,
        overrides={
            "bins": {"n_bins": 3},
            "uncertainties": {"scatter_scale": 0.08},
        },
    )

    assert result.n_bins == 3
    assert result.spec is not None
    assert result.spec["bins"]["n_bins"] == 3
    assert result.spec["uncertainties"]["scatter_scale"] == 0.08


def test_overrides_preserve_other_fields(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that nested overrides preserve unspecified fields."""
    result = build_survey_tomography(
        cfg=photoz_config,
        overrides={
            "uncertainties": {"scatter_scale": 0.08},
        },
    )

    assert result.spec is not None
    assert result.spec["uncertainties"]["scatter_scale"] == 0.08
    assert result.spec["uncertainties"]["mean_offset"] == 0.01


def test_configuration_not_mutated(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that building tomography does not mutate the configuration."""
    original = deepcopy(photoz_config)

    build_survey_tomography(
        cfg=photoz_config,
        overrides={
            "uncertainties": {"scatter_scale": 0.08},
        },
    )

    assert photoz_config == original


def test_survey_metadata(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that survey metadata is attached only when requested."""
    without = build_survey_tomography(cfg=photoz_config)

    with_metadata = build_survey_tomography(
        cfg=photoz_config,
        include_survey_metadata=True,
    )

    assert without.survey_metadata == {}
    assert isinstance(with_metadata.survey_metadata, dict)
    assert with_metadata.survey_metadata


@pytest.mark.parametrize(
    ("include_metadata", "include_diagnostics"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_photoz_metadata_options(
    photoz_config: dict[str, Any],
    include_metadata: bool,
    include_diagnostics: bool,
) -> None:
    """Tests that photo-z metadata and diagnostics work independently."""
    result = build_survey_tomography(
        cfg=photoz_config,
        include_tomo_metadata=include_metadata,
        include_photoz_diagnostics=include_diagnostics,
    )

    assert isinstance(result.metadata, dict)
    assert result.n_bins == 4

    if include_metadata or include_diagnostics:
        assert result.metadata
    else:
        assert result.metadata == {}


@pytest.mark.parametrize(
    "include_metadata",
    [False, True],
)
def test_specz_metadata(
    specz_config: dict[str, Any],
    include_metadata: bool,
) -> None:
    """Tests that spec-z tomography supports optional numerical metadata."""
    result = build_survey_tomography(
        cfg=specz_config,
        include_tomo_metadata=include_metadata,
    )

    assert isinstance(result.metadata, dict)

    if include_metadata:
        assert result.metadata
    else:
        assert result.metadata == {}


@pytest.mark.parametrize("include_metadata", [False, True])
def test_predefined_metadata(include_metadata: bool) -> None:
    """Tests that predefined tomography supports optional metadata."""
    result = build_survey_tomography(
        config_file="mock_predefined_config.yaml",
        include_tomo_metadata=include_metadata,
    )

    assert result.spec is not None
    assert result.spec["kind"] == "predefined"
    assert result.n_bins == 5
    assert isinstance(result.metadata, dict)


def test_predefined_matches_direct_configuration() -> None:
    """Tests that predefined tomography is stable across repeated builds."""
    first = build_survey_tomography(
        config_file="mock_predefined_config.yaml",
    )
    second = build_survey_tomography(
        config_file="mock_predefined_config.yaml",
    )

    np.testing.assert_allclose(first.z, second.z)
    np.testing.assert_allclose(first.nz, second.nz)

    assert first.bins.keys() == second.bins.keys()

    for index in first.bins:
        np.testing.assert_allclose(first.bins[index], second.bins[index])


def test_role_and_year_selection(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that role and year selectors identify the requested entry."""
    result = build_survey_tomography(
        cfg=photoz_config,
        role="lens",
        year=1,
    )

    assert result.spec is not None
    assert result.spec["role"] == "lens"
    assert str(result.spec["year"]) == "1"


def test_ambiguous_selection(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that ambiguous survey selections raise an error."""
    cfg = deepcopy(photoz_config)

    second = deepcopy(cfg["tomography"][0])
    second["role"] = "source"
    cfg["tomography"].append(second)

    with pytest.raises(ValueError):
        build_survey_tomography(cfg=cfg)


def test_unmatched_selection(
    photoz_config: dict[str, Any],
) -> None:
    """Tests that unmatched selectors raise an error."""
    with pytest.raises(ValueError):
        build_survey_tomography(
            cfg=photoz_config,
            role="source",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "survey": "lsst",
            "config_file": "example_minimal_photoz.yaml",
        },
        {
            "survey": "lsst",
            "cfg": {},
        },
        {
            "config_file": "example_minimal_photoz.yaml",
            "cfg": {},
        },
    ],
)
def test_invalid_configuration_sources(
    kwargs: dict[str, Any],
) -> None:
    """Tests that exactly one configuration source must be supplied."""
    with pytest.raises(ValueError, match="exactly one"):
        build_survey_tomography(**kwargs)
