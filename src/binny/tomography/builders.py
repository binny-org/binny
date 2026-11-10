"""High-level functional tomography builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import binny.surveys.config_utils as cu
from binny.binning.photoz import build_photoz_bins
from binny.binning.specz import build_specz_bins
from binny.tomography.core import Tomography
from binny.tomography.predefined import build_predefined_tomography

__all__ = ["build_tomography_from_spec"]


def build_tomography_from_spec(
    z: Any | None = None,
    nz: Any | None = None,
    *,
    tomo_spec: Mapping[str, Any],
    predefined_bins: Mapping[int, Any] | None = None,
    include_tomo_metadata: bool = False,
    include_photoz_diagnostics: bool = False,
) -> Tomography:
    """Build tomography from a survey-style specification.

    Args:
        z: True-redshift grid. Required for photo-z and spec-z tomography,
            and when predefined bins are supplied directly.
        nz: Parent redshift distribution. Optional for predefined bins.
        tomo_spec: Tomography specification defining the binning model,
            bin edges, and uncertainties.
        predefined_bins: Optional mapping of externally supplied bin
            distributions for predefined tomography.
        include_tomo_metadata: Whether to include numerical binning metadata.
        include_photoz_diagnostics: Whether to include photo-z assignment
            diagnostics in the metadata.

    Returns:
        Tomography containing the constructed bins, optional numerical
        metadata, and the resolved tomography specification.

    Raises:
        ValueError: If required arrays are missing or the specification
            is invalid.
    """
    spec_in = dict(tomo_spec)
    spec_in.setdefault("nz", {"model": "arrays"})
    spec = cu._parse_entry(spec_in)

    kind = spec["kind"]

    if kind == "predefined":
        if predefined_bins is None:
            z_arr, predefined_bins = cu._load_predefined_bin_source(spec["bins"])
        else:
            if z is None:
                raise ValueError("z is required when predefined_bins is provided.")
            z_arr = np.asarray(z, dtype=float)

        result = build_predefined_tomography(
            z_arr,
            predefined_bins,
            nz=nz,
            normalize=spec["bins"].get("normalize_bins", False),
            norm_method=spec["bins"].get("norm_method", "trapezoid"),
            include_metadata=include_tomo_metadata,
        )
    else:
        if z is None or nz is None:
            raise ValueError("z and nz are required for photoz/specz tomography.")

        if kind == "photoz":
            builder = build_photoz_bins
        elif kind == "specz":
            builder = build_specz_bins
        else:
            raise ValueError(f"Unknown tomography kind {kind!r}.")

        kwargs = cu._builder_kwargs_from_spec(spec)

        if kind == "photoz":
            kwargs["include_photoz_diagnostics"] = include_photoz_diagnostics

        metadata_requested = include_tomo_metadata or (
            kind == "photoz" and include_photoz_diagnostics
        )

        output = builder(
            z=z,
            nz=nz,
            include_metadata=metadata_requested,
            **kwargs,
        )

        if metadata_requested:
            bins, metadata = output
        else:
            bins, metadata = output, {}

        result = Tomography(
            z=np.asarray(z, dtype=float),
            nz=np.asarray(nz, dtype=float),
            bins=bins,
            metadata=metadata if metadata is not None else {},
        )

    result.metadata = result.metadata if result.metadata is not None else {}
    result.spec = dict(spec)

    return result
