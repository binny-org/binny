"""LSST tomography convenience wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from binny.nz_tomo.photoz import build_photoz_bins
from binny.surveys.config_core import survey_from_mapping
from binny.utils.io import load_yaml

Sample = Literal["lens", "source"]

__all__ = ["lsst_tomography"]


def _year_key(year: int) -> str:
    """Converts year selector to YAML key."""
    if year in (1, 10):
        return f"y{year}"
    raise ValueError("year must be 1 or 10 for LSST presets.")


def lsst_tomography(
    *,
    year: int,
    sample: Sample,
    z: Any | None = None,
    config_file: str = "lsst_survey_specs.yaml",
    include_survey_metadata: bool = False,
    include_tomo_metadata: bool = False,
):
    """Builds LSST photo-z tomographic bins from the shipped LSST YAML
    preset."""
    yk = _year_key(year)

    cfg_all = load_yaml(config_file, package="binny.surveys.configs")
    cfg_lsst = cfg_all.get("lsst")
    if not isinstance(cfg_lsst, Mapping):
        raise ValueError("Preset YAML must contain top-level mapping 'lsst'.")

    sample_keys = {
        "lens": "lens_sample",
        "source": "source_sample",
    }

    sample_key = sample_keys[sample]
    sample_block = cfg_lsst.get(sample_key)
    if not isinstance(sample_block, Mapping):
        raise ValueError(f"Preset missing mapping lsst.{sample_key}.")

    block = sample_block.get(yk)
    if not isinstance(block, Mapping):
        raise ValueError(f"Preset missing mapping lsst.{sample_key}.{yk}.")

    grid = cfg_lsst.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("Preset missing mapping lsst.grid.")

    sm = block.get("smail")
    if not isinstance(sm, Mapping):
        raise ValueError(f"Preset missing mapping lsst.{sample_key}.{yk}.smail.")

    pz = block.get("photoz")
    if not isinstance(pz, Mapping):
        raise ValueError(f"Preset missing mapping lsst.{sample_key}.{yk}.photoz.")

    try:
        n_bins = int(block["n_tomo_bins"])
        z0 = float(sm["z0"])
        alpha = float(sm["alpha"])
        beta = float(sm["beta"])
        scatter_scale = float(pz["scatter_scale"])
        mean_offset = float(pz["mean_offset"])
    except KeyError as e:
        raise ValueError(f"Preset missing required key: {e.args[0]!r}.") from e

    default_scheme = "equidistant" if sample == "lens" else "equal_number"
    scheme = str(block.get("binning", default_scheme)).lower()

    # Build a minimal in-memory config entry and reuse the generic parser.
    cfg_entry: dict[str, Any] = {
        "name": str(cfg_lsst.get("name", "lsst")),
        "grid": grid,
        "footprint": cfg_lsst.get("footprint"),
        "nz": {
            "model": "smail",
            "params": {"z0": z0, "alpha": alpha, "beta": beta},
        },
        "tomo": {
            "kind": "photoz",
            "binning_scheme": scheme,
            "params": {
                "n_bins": n_bins,
                "scatter_scale": scatter_scale,
                "mean_offset": mean_offset,
            },
        },
    }

    z_arr, nz_arr, tomo, survey_meta = survey_from_mapping(
        cfg=cfg_entry,
        key="lsst",
        z=z,
        include_survey_metadata=include_survey_metadata,
    )

    bins_out = build_photoz_bins(
        z=z_arr,
        nz=nz_arr,
        bin_edges=tomo["bin_edges"],
        binning_scheme=tomo["binning_scheme"],
        include_metadata=include_tomo_metadata,
        **tomo["params"],
    )

    if include_tomo_metadata:
        bins, tomo_meta = bins_out
    else:
        bins, tomo_meta = bins_out, None

    if include_survey_metadata and include_tomo_metadata:
        return bins, survey_meta, tomo_meta
    if include_survey_metadata:
        return bins, survey_meta
    if include_tomo_metadata:
        return bins, tomo_meta
    return bins
