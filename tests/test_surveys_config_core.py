"""Unit tests for ``binny.surveys.config_core`` module."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files
from typing import Any

import numpy as np
import pytest
import yaml

from binny.surveys.config_core import (
    build_from_arrays,
    build_from_config,
    build_from_mapping,
    spec_from_config,
    spec_from_mapping,
    survey_from_config,
    survey_from_mapping,
)


@pytest.fixture
def minimal_mapping() -> dict[str, Any]:
    """Tests that a minimal new-schema mapping is valid."""
    return {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "uncertainties": {"scatter_scale": 0.05},
            }
        ],
    }


def test_survey_from_mapping_returns_expected_shapes(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that survey_from_mapping returns z and nz with matching shape."""
    z, nz, spec, meta = survey_from_mapping(
        cfg=minimal_mapping,
        role="lens",
        year="1",
        include_survey_metadata=True,
    )

    assert isinstance(z, np.ndarray)
    assert isinstance(nz, np.ndarray)
    assert z.shape == nz.shape
    assert isinstance(spec, dict)
    assert isinstance(meta, dict)


def test_survey_from_mapping_returns_none_meta_when_not_requested(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that survey_from_mapping returns meta=None when not requested."""
    z, nz, spec, meta = survey_from_mapping(
        cfg=minimal_mapping,
        role="lens",
        year="1",
        include_survey_metadata=False,
    )
    assert z.shape == nz.shape
    assert isinstance(spec, dict)
    assert meta is None


def test_survey_from_config_loads_shipped_examples() -> None:
    """Tests that shipped example configs load via survey_from_config."""
    pkg = files("binny.surveys.configs")

    for name in [
        "example_minimal_photoz.yaml",
        "example_full_photoz.yaml",
        "example_minimal_specz.yaml",
        "example_full_specz.yaml",
    ]:
        with as_file(pkg / name) as path:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert isinstance(raw, Mapping)

            for entry in raw["tomography"]:
                z, nz, spec, _ = survey_from_config(
                    config_file=path,
                    role=str(entry["role"]),
                    year=str(entry["year"]),
                )
                assert z.shape == nz.shape
                assert isinstance(spec, dict)


def test_spec_from_mapping_enforces_kind(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that spec_from_mapping rejects mismatched kinds."""
    with pytest.raises(ValueError, match="kind must be"):
        spec_from_mapping(
            kind="specz",
            cfg=minimal_mapping,
            role="lens",
            year="1",
        )


def test_spec_from_config_returns_expected_tuple(
    minimal_mapping: Mapping[str, Any],
    tmp_path,
) -> None:
    """Tests that spec_from_config returns (z, nz, spec[, meta])."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(minimal_mapping), encoding="utf-8")

    z, nz, spec, meta = spec_from_config(
        kind="photoz",
        config_file=path,
        role="lens",
        year="1",
        include_survey_metadata=True,
    )

    assert isinstance(z, np.ndarray)
    assert isinstance(nz, np.ndarray)
    assert isinstance(spec, dict)
    assert isinstance(meta, dict)


def test_spec_from_config_enforces_kind(tmp_path) -> None:
    """Tests that spec_from_config rejects mismatched kinds."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
            }
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=r"kind must be"):
        spec_from_config(kind="specz", config_file=path, role="lens", year="1")


def test_build_from_mapping_calls_builder(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that build_from_mapping wires parsed inputs to the builder."""

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that asserts wiring and returns a single bin."""
        assert isinstance(z, np.ndarray)
        assert isinstance(nz, np.ndarray)

        assert bin_edges is None
        assert binning_scheme == "equidistant"
        assert n_bins == 2
        assert include_metadata is False

        # ensure extra params from uncertainties are forwarded
        assert "scatter_scale" in params

        return {0: nz.copy()}

    bins = build_from_mapping(
        kind="photoz",
        builder=dummy_builder,
        cfg=minimal_mapping,
        role="lens",
        year="1",
    )

    assert isinstance(bins, dict)
    assert 0 in bins
    assert isinstance(bins[0], np.ndarray)


def test_build_from_mapping_returns_survey_meta_only(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that build_from_mapping can return (bins, survey_meta) only."""

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that returns bins only when include_metadata is False."""
        assert include_metadata is False
        return {0: nz.copy()}

    bins, survey_meta = build_from_mapping(
        kind="photoz",
        builder=dummy_builder,
        cfg=minimal_mapping,
        role="lens",
        year="1",
        include_survey_metadata=True,
        include_tomo_metadata=False,
    )

    assert isinstance(bins, dict)
    assert isinstance(survey_meta, dict)


def test_build_from_mapping_supports_explicit_edges_and_tomo_meta(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that build_from_mapping handles explicit edges and tomo metadata."""
    cfg = dict(minimal_mapping)
    cfg["tomography"] = [
        {
            "role": "lens",
            "year": "1",
            "kind": "photoz",
            "nz": cfg["tomography"][0]["nz"],
            "bins": {"edges": [0.0, 0.3, 1.0]},
            "uncertainties": {"scatter_scale": 0.05},
        }
    ]

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that checks edge wiring and returns meta."""
        assert isinstance(z, np.ndarray)
        assert isinstance(nz, np.ndarray)

        assert np.allclose(np.asarray(bin_edges, dtype=float), [0.0, 0.3, 1.0])
        assert binning_scheme is None
        assert n_bins is None

        assert include_metadata is True
        assert params["scatter_scale"] == pytest.approx(0.05)
        assert "bin_range" not in params

        bins = {0: nz.copy()}
        tomo_meta = {"bin_edges": np.asarray(bin_edges, dtype=float)}
        return bins, tomo_meta

    bins, tomo_meta = build_from_mapping(
        kind="photoz",
        builder=dummy_builder,
        cfg=cfg,
        role="lens",
        year="1",
        include_tomo_metadata=True,
        include_survey_metadata=False,
    )

    assert isinstance(bins, dict)
    assert isinstance(tomo_meta, dict)


def test_build_from_mapping_returns_both_metas_when_requested(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that build_from_mapping returns both survey and tomo metadata."""

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder returning (bins, tomo_meta) when include_metadata."""
        assert include_metadata is True
        return {0: nz.copy()}, {"tomo": "meta"}

    bins, survey_meta, tomo_meta = build_from_mapping(
        kind="photoz",
        builder=dummy_builder,
        cfg=minimal_mapping,
        role="lens",
        year="1",
        include_survey_metadata=True,
        include_tomo_metadata=True,
    )

    assert isinstance(bins, dict)
    assert isinstance(survey_meta, dict)
    assert tomo_meta == {"tomo": "meta"}


def test_build_from_arrays_works_without_config() -> None:
    """Tests that build_from_arrays works with in-memory inputs."""
    z = np.linspace(0.0, 1.0, 11)
    nz = np.exp(-z)

    tomo_spec = {
        "role": "lens",
        "year": "1",
        "kind": "photoz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"scheme": "equidistant", "n_bins": 2},
        "uncertainties": {},
    }

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that accepts wrapper kwargs and returns a single bin."""
        assert isinstance(z, np.ndarray)
        assert isinstance(nz, np.ndarray)

        assert bin_edges is None
        assert binning_scheme == "equidistant"
        assert n_bins == 2

        assert include_metadata is False
        assert params == {}

        return {0: nz}

    bins = build_from_arrays(
        kind="photoz",
        builder=dummy_builder,
        z=z,
        nz=nz,
        tomo_spec=tomo_spec,
    )

    assert isinstance(bins, dict)
    assert 0 in bins


def test_build_from_arrays_returns_tomo_meta_when_requested() -> None:
    """Tests that build_from_arrays returns (bins, tomo_meta) when requested."""
    z = np.linspace(0.0, 1.0, 11)
    nz = np.exp(-0.5 * z)

    tomo_spec = {
        "role": "lens",
        "year": "1",
        "kind": "photoz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"scheme": "equidistant", "n_bins": 2, "range": [0.0, 1.0]},
        "uncertainties": {},
    }

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that checks bin_range injection and returns meta."""
        assert include_metadata is True
        assert tuple(params["bin_range"]) == (0.0, 1.0)
        return {0: nz.copy()}, {"tomo": "meta"}

    bins, tomo_meta = build_from_arrays(
        kind="photoz",
        builder=dummy_builder,
        z=z,
        nz=nz,
        tomo_spec=tomo_spec,
        include_tomo_metadata=True,
    )

    assert isinstance(bins, dict)
    assert tomo_meta == {"tomo": "meta"}


def test_build_from_config_calls_builder(tmp_path) -> None:
    """Tests that build_from_config wires parsed inputs from YAML to the builder."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "uncertainties": {"scatter_scale": 0.05},
            }
        ],
    }

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges,
        binning_scheme,
        n_bins,
        include_metadata=False,
        **params,
    ):
        """Mock builder that just returns nz."""
        assert isinstance(z, np.ndarray)
        assert isinstance(nz, np.ndarray)
        assert z.shape == nz.shape
        assert bin_edges is None
        assert binning_scheme == "equidistant"
        assert n_bins == 2
        assert include_metadata is False
        assert "scatter_scale" in params
        return {0: nz.copy()}

    bins = build_from_config(
        kind="photoz",
        builder=dummy_builder,
        config_file=path,
        role="lens",
        year="1",
    )

    assert isinstance(bins, dict)
    assert 0 in bins
    assert isinstance(bins[0], np.ndarray)


def test_build_from_config_returns_tomo_meta_when_requested(tmp_path) -> None:
    """Tests that build_from_config returns (bins, tomo_meta) when requested."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
            }
        ],
    }

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that returns (bins, meta) when include_metadata."""
        assert include_metadata is True
        return {0: nz.copy()}, {"tomo": "meta"}

    bins, tomo_meta = build_from_config(
        kind="photoz",
        builder=dummy_builder,
        config_file=path,
        role="lens",
        year="1",
        include_tomo_metadata=True,
    )

    assert isinstance(bins, dict)
    assert tomo_meta == {"tomo": "meta"}


def test_survey_from_mapping_rejects_non_mapping() -> None:
    """Tests that survey_from_mapping rejects non-mapping cfg inputs."""
    with pytest.raises(ValueError, match=r"cfg must be a mapping"):
        survey_from_mapping(cfg=123, role="lens", year="1")


def test_spec_from_mapping_returns_three_tuple_by_default(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that spec_from_mapping returns (z, nz, spec) by default."""
    z, nz, spec = spec_from_mapping(kind="photoz", cfg=minimal_mapping, role="lens", year="1")
    assert z.shape == nz.shape
    assert isinstance(spec, dict)


def test_build_from_config_returns_survey_meta_only(tmp_path) -> None:
    """Tests that build_from_config can return (bins, survey_meta) only."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
            }
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that returns bins only."""
        assert include_metadata is False
        return {0: nz.copy()}

    bins, survey_meta = build_from_config(
        kind="photoz",
        builder=dummy_builder,
        config_file=path,
        role="lens",
        year="1",
        include_survey_metadata=True,
        include_tomo_metadata=False,
    )
    assert isinstance(bins, dict)
    assert isinstance(survey_meta, dict)


def test_build_from_config_returns_both_metas_when_requested(tmp_path) -> None:
    """Tests that build_from_config returns (bins, survey_meta, tomo_meta) when requested."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"scheme": "equidistant", "n_bins": 2},
            }
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder returning (bins, tomo_meta) when include_metadata."""
        assert include_metadata is True
        return {0: nz.copy()}, {"tomo": "meta"}

    bins, survey_meta, tomo_meta = build_from_config(
        kind="photoz",
        builder=dummy_builder,
        config_file=path,
        role="lens",
        year="1",
        include_survey_metadata=True,
        include_tomo_metadata=True,
    )
    assert isinstance(bins, dict)
    assert isinstance(survey_meta, dict)
    assert tomo_meta == {"tomo": "meta"}


def test_build_from_config_supports_explicit_edges(tmp_path) -> None:
    """Tests that build_from_config wires explicit edges into the builder."""
    cfg = {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {
                    "model": "smail",
                    "params": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
                },
                "bins": {"edges": [0.0, 0.4, 1.0]},
            }
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        """Mock builder that asserts explicit edges path."""
        assert np.allclose(np.asarray(bin_edges, dtype=float), [0.0, 0.4, 1.0])
        assert binning_scheme is None
        assert n_bins is None
        return {0: nz.copy()}

    bins = build_from_config(
        kind="photoz",
        builder=dummy_builder,
        config_file=path,
        role="lens",
        year="1",
    )
    assert isinstance(bins, dict)


def test_survey_from_config_ignores_key_argument(
    minimal_mapping: Mapping[str, Any],
    tmp_path,
) -> None:
    """Tests that survey_from_config ignores key in the flat schema."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(minimal_mapping), encoding="utf-8")

    z1, nz1, spec1, meta1 = survey_from_config(config_file=path, role="lens", year="1")
    z2, nz2, spec2, meta2 = survey_from_config(
        config_file=path,
        key="alt",  # currently ignored in survey_from_config
        role="lens",
        year="1",
    )

    assert np.allclose(z1, z2)
    assert np.allclose(nz1, nz2)
    assert spec1 == spec2
    assert meta1 is None
    assert meta2 is None


def test_survey_from_mapping_raises_on_no_matches(minimal_mapping) -> None:
    """Tests that survey_from_mapping raises when no tomography entry matches."""
    with pytest.raises(ValueError, match="tomography entry"):
        survey_from_mapping(cfg=minimal_mapping, role="source", year="1")


def test_survey_from_mapping_raises_on_multiple_matches(
    minimal_mapping,
) -> None:
    """Tests that survey_from_mapping raises when multiple entries match."""
    cfg = dict(minimal_mapping)
    cfg["tomography"] = [cfg["tomography"][0], dict(cfg["tomography"][0])]
    with pytest.raises(ValueError, match="tomography entry"):
        survey_from_mapping(cfg=cfg, role="lens", year="1")


def test_kind_defaults_to_photoz(minimal_mapping) -> None:
    """Tests that missing kind defaults to photoz."""
    cfg = dict(minimal_mapping)
    entry = dict(cfg["tomography"][0])
    entry.pop("kind", None)
    cfg["tomography"] = [entry]

    z, nz, spec = spec_from_mapping(kind="photoz", cfg=cfg, role="lens", year="1")
    assert spec["kind"] == "photoz"
    assert z.shape == nz.shape


def test_build_from_mapping_injects_bin_range(
    minimal_mapping: Mapping[str, Any],
) -> None:
    """Tests that bins.range is forwarded as bin_range to the builder."""
    cfg = dict(minimal_mapping)
    cfg["tomography"] = [
        {
            **cfg["tomography"][0],
            "bins": {"scheme": "equidistant", "n_bins": 2, "range": [0.1, 0.9]},
        }
    ]

    def dummy_builder(
        *,
        z,
        nz,
        bin_edges=None,
        binning_scheme=None,
        n_bins=None,
        include_metadata=False,
        **params,
    ):
        assert tuple(params["bin_range"]) == (0.1, 0.9)
        return {0: nz.copy()}

    bins = build_from_mapping(
        kind="photoz",
        builder=dummy_builder,
        cfg=cfg,
        role="lens",
        year="1",
    )
    assert 0 in bins
