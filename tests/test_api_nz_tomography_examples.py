"""Tests that NZTomography loads and builds bins from schema YAML examples."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files
from typing import Any

import numpy as np
import pytest
import yaml

from binny.api.nz_tomography import NZTomography


@pytest.mark.parametrize(
    "fname",
    [
        "example_minimal_photoz.yaml",
        "example_full_photoz.yaml",
        "example_minimal_specz.yaml",
        "example_full_specz.yaml",
        "example_tabulated.yaml",
        "mock_predefined_config.yaml",
    ],
)
def test_examples_load_and_build_bins(
    fname: str,
) -> None:
    """Tests that schema YAML examples load and build bins successfully."""
    config_resource = files("binny.surveys.configs").joinpath(fname)

    with as_file(config_resource) as path:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert isinstance(
            raw,
            Mapping,
        )

        root = (
            raw["survey"]
            if isinstance(
                raw.get("survey"),
                Mapping,
            )
            else raw
        )

        assert "tomography" in root
        assert isinstance(
            root["tomography"],
            list,
        )
        assert root["tomography"]

        for entry in root["tomography"]:
            assert isinstance(
                entry,
                Mapping,
            )

            kind = (
                str(
                    entry.get(
                        "kind",
                        "photoz",
                    )
                )
                .strip()
                .lower()
            )

            assert kind in {
                "photoz",
                "specz",
                "predefined",
            }

            if kind != "predefined":
                assert "z_grid" in root
                assert isinstance(
                    entry.get("nz"),
                    Mapping,
                )

            bins_spec = entry.get("bins")

            assert isinstance(
                bins_spec,
                Mapping,
            )

            uncertainties = entry.get("uncertainties")

            if uncertainties is not None:
                assert isinstance(
                    uncertainties,
                    Mapping,
                )

            selectors: dict[str, Any] = {}

            for selector in (
                "role",
                "year",
                "scenario",
                "sample",
            ):
                value = entry.get(selector)

                if value is not None:
                    selectors[selector] = str(value)

            tomography = NZTomography()

            result = tomography.build_bins(
                config_file=path,
                key=None,
                include_survey_metadata=True,
                include_tomo_metadata=True,
                **selectors,
            )

            assert result.spec["kind"] == kind

            z = result.z
            nz = result.nz
            built_bins = result.bins

            assert isinstance(
                z,
                np.ndarray,
            )
            assert z.ndim == 1
            assert z.size >= 2
            assert np.all(np.isfinite(z))
            assert np.all(np.diff(z) > 0.0)

            assert isinstance(
                nz,
                np.ndarray,
            )
            assert nz.shape == z.shape
            assert np.all(np.isfinite(nz))
            assert np.all(nz >= 0.0)

            assert isinstance(
                built_bins,
                Mapping,
            )
            assert len(built_bins) >= 1

            for bin_index, curve in built_bins.items():
                assert isinstance(
                    bin_index,
                    int,
                )
                assert isinstance(
                    curve,
                    np.ndarray,
                )
                assert curve.shape == z.shape
                assert np.all(np.isfinite(curve))
                assert np.all(curve >= 0.0)

            shape_statistics = result.shape_stats()

            assert isinstance(
                shape_statistics,
                dict,
            )
            assert set(shape_statistics["per_bin"]) == set(built_bins)

            population_statistics = result.population_stats()

            assert isinstance(
                population_statistics,
                dict,
            )
            assert set(population_statistics["fractions"]) == set(built_bins)
