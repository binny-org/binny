"""Survey-level convenience APIs.

This module provides a set of high-level helpers that return commonly used
survey products (e.g., tomographic redshift bins) with a stable, user-facing
interface.
"""

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
    """Build LSST tomographic redshift bins for a given sample and survey year.

    This function is a public API wrapper around
    :func:`binny.surveys.lsst.lsst_tomography`. It returns a dictionary mapping
    tomographic bin indices to per-bin redshift distributions evaluated on the
    provided redshift grid.

    Args:
        z: Redshift grid on which the distributions are defined. Any array-like
            input is accepted and forwarded to the underlying implementation.
        year: LSST survey year/scenario selector (e.g., 1 or 10).
        sample: Which LSST sample to build bins for. Use ``"source"`` for
            weak-lensing source galaxies or ``"lens"`` for number-count lens
            galaxies.
        config_file: Name or path of the LSST survey specification YAML file
            used by the underlying implementation.
        normalize_input: If ``True``, normalize the input redshift distribution
            before binning (behavior defined by the underlying implementation).
        normalize_bins: If ``True``, normalize each tomographic bin distribution
            after binning (behavior defined by the underlying implementation).

    Returns:
        Dictionary mapping bin index (``int``) to a NumPy array of values
        sampled on ``z``. Arrays have the same shape as ``z``.

    Raises:
        ValueError: If ``sample`` is not one of ``"lens"`` or ``"source"``, or if
            ``year`` is not supported by the LSST configuration.
        OSError: If the configuration file cannot be read.
        yaml.YAMLError: If the configuration file is not valid YAML.

    Examples:
        >>> import numpy as np
        >>> from binny.api.surveys import lsst_tomography
        >>> z = np.linspace(0.0, 3.0, 11)
        >>> bins = lsst_tomography(z=z, year=1, sample="source")
        >>> isinstance(bins, dict)
        True
    """
    return _lsst_tomography(
        z=z,
        year=year,
        sample=sample,
        config_file=config_file,
        normalize_input=normalize_input,
        normalize_bins=normalize_bins,
    )
