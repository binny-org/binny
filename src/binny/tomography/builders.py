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


def build_tomography_from_spec(
    z: Any | None = None,
    nz: Any | None = None,
    *,
    tomo_spec: Mapping[str, Any],
    predefined_bins: Mapping[int, Any] | None = None,
    include_tomo_metadata: bool = False,
    include_photoz_diagnostics: bool = False,
) -> Tomography:
    """Build tomography from a survey-style specification."""
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

        builder = build_photoz_bins if kind == "photoz" else build_specz_bins

        kwargs = cu._builder_kwargs_from_spec(spec)

        if kind == "photoz":
            kwargs["include_photoz_diagnostics"] = include_photoz_diagnostics

        metadata_requested = include_tomo_metadata or (
            kind == "photoz" and include_photoz_diagnostics
        )

        output = builder(
            z=z,
            nz=nz,
            include_metadata=include_tomo_metadata,
            **kwargs,
        )

        if metadata_requested:
            bins, metadata = output
        else:
            bins, metadata = output, None

        result = Tomography(
            z=np.asarray(z, dtype=float),
            nz=np.asarray(nz, dtype=float),
            bins=bins,
            metadata=metadata,
        )

    return result
