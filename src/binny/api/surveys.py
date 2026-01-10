# src/binny/api/surveys.py
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from binny.surveys.lsst import lsst_tomography as _lsst_tomography

Sample = Literal["lens", "source"]

__all__ = ["lsst_tomography"]


def lsst_tomography(
    *,
    z: Any,
    year: int,
    sample: Sample,
    config_file: str = "lsst_survey_specs.yaml",
    normalize_input: bool = True,
    normalize_bins: bool = True,
) -> dict[int, np.ndarray]:
    """API wrapper around :func:`binny.surveys.lsst.lsst_tomography`."""
    return _lsst_tomography(
        z=z,
        year=year,
        sample=sample,
        config_file=config_file,
        normalize_input=normalize_input,
        normalize_bins=normalize_bins,
    )
