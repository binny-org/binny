"""Tomography configurations for LSST survey samples."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from binny.api.distributions import redshift_distribution
from binny.api.edges import bin_edges
from binny.api.tomo import photoz_bins
from binny.utils.io import load_yaml

Sample = Literal["lens", "source"]

__all__ = ["lsst_tomography"]


def _year_key(year: int) -> str:
    if year in (1, 10):
        return f"y{year}"
    raise ValueError("year must be 1 or 10 for this LSST preset.")


def lsst_tomography(
    *,
    z: Any,
    year: int,
    sample: Sample,
    config_file: str = "lsst_survey_specs.yaml",
    normalize_input: bool = True,
    normalize_bins: bool = True,
) -> dict[int, np.ndarray]:
    """Builds LSST photo-z tomographic n_i(z) from a packaged YAML preset."""
    cfg_all = load_yaml(config_file, package="binny.surveys.configs")

    try:
        cfg = cfg_all["lsst"]
    except KeyError as e:
        raise ValueError("Preset YAML must contain top-level key 'lsst'.") from e

    yk = _year_key(year)

    sample_key = "lens_sample" if sample == "lens" else "source_sample"
    try:
        block = cfg[sample_key][yk]
    except KeyError as e:
        raise ValueError(f"Preset missing {sample_key}.{yk}.") from e

    # Parent n(z)
    try:
        sm = block["smail"]
        z0 = float(sm["z0"])
        alpha = float(sm["alpha"])
        beta = float(sm["beta"])
    except KeyError as e:
        raise ValueError(
            f"Preset missing required smail parameter: {e.args[0]}."
        ) from e

    nz = redshift_distribution("smail", z, z0=z0, alpha=alpha, beta=beta)

    # Tomographic edges
    try:
        n_bins = int(block["n_tomo_bins"])
    except KeyError as e:
        raise ValueError("Preset missing required key 'n_tomo_bins'.") from e

    default_method = "equidistant" if sample == "lens" else "equal_number"
    method = str(block.get("binning", default_method))

    if method == "equidistant":
        # equidistant uses x_min/x_max only
        if "bin_range" in block:
            x_min, x_max = map(float, block["bin_range"])
        else:
            x_min, x_max = float(np.min(z)), float(np.max(z))

        edges = bin_edges("equidistant", x_min=x_min, x_max=x_max, n_bins=n_bins)

    elif method == "equal_number":
        # equal_number uses x, weights
        edges = bin_edges("equal_number", x=z, weights=nz, n_bins=n_bins)

    else:
        raise ValueError(
            f"Unsupported binning='{method}' for LSST preset. "
            "Use 'equidistant' (lens) or 'equal_number' (source)."
        )

    # Photo-z model
    try:
        pz = block["photoz"]
        scatter_scale = float(pz["scatter_scale"])
        mean_offset = float(pz["mean_offset"])
    except KeyError as e:
        raise ValueError(
            f"Preset missing required photoz parameter: {e.args[0]}."
        ) from e

    return photoz_bins(
        z=z,
        nz=nz,
        bin_edges=edges,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        normalize_input=normalize_input,
        normalize_bins=normalize_bins,
    )
