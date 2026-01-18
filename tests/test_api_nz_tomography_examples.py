"""Tests that NZTomography loads and builds bins from schema YAML examples."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files

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
    ],
)
def test_examples_load_and_build_bins(fname: str) -> None:
    """Tests that schema YAML examples load and build bins successfully."""
    cfg = files("binny.surveys.configs").joinpath(fname)
    with as_file(cfg) as path:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, Mapping)

        if "survey" in raw and isinstance(raw["survey"], Mapping):
            root = raw["survey"]
        else:
            root = raw

        assert "z_grid" in root
        assert "tomography" in root
        assert isinstance(root["tomography"], list)
        assert root["tomography"]

        for entry in root["tomography"]:
            assert isinstance(entry, Mapping)

            # Required selectors
            role = entry.get("role")
            year = entry.get("year")
            assert role is not None
            assert year is not None

            # Optional numeric metadata must be None or numeric
            if "n_gal_arcmin2" in entry:
                val = entry.get("n_gal_arcmin2")
                assert val is None or isinstance(val, int | float)

            # Uncertainties must be a mapping if present
            unc = entry.get("uncertainties")
            if unc is not None:
                assert isinstance(unc, Mapping)

            t = NZTomography.from_config(
                config_file=path,
                key=None,
                role=str(role),
                year=str(year),
                include_survey_metadata=True,
            )

            _ = t.build(include_metadata=True)

            z = t.z()
            bins = t.bins()

            assert isinstance(z, np.ndarray)
            assert z.ndim == 1
            assert isinstance(bins, Mapping)
            assert len(bins) >= 1

            for k, v in bins.items():
                assert isinstance(k, int)
                assert isinstance(v, np.ndarray)
                assert v.shape == z.shape
                assert np.all(np.isfinite(v))

            # Shape stats must always work
            out = t.shape_stats()
            assert isinstance(out, dict)

            # Population stats must work when metadata is requested
            pop = t.population_stats()
            assert isinstance(pop, dict)
