"""Tomographic binning API.

This module provides a public interface for constructing tomographic
redshift bins from a parent redshift distribution. It supports both:

- **Photo-z binning** (with optional photometric redshift scatter / outliers)
- **Spec-z binning** (with optional spectroscopic selection effects)

The underlying implementations live in :mod:`binny.ztomo.photoz` and
:mod:`binny.ztomo.specz`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from binny.ztomo.photoz import build_photoz_bins
from binny.ztomo.specz import build_specz_bins

__all__ = [
    "photoz_bins",
    "specz_bins",
    "tomo_bins",
]


def photoz_bins(z: Any, nz: Any, bin_edges: Any, **params: Any):
    """Builds tomographic bins for a photometric redshift (photo-z) sample.

    This is a public API wrapper around :func:`binny.ztomo.photoz.build_photoz_bins`.
    It returns a mapping of tomographic bin indices to per-bin redshift
    distributions evaluated on the input grid.

    Args:
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Bin edges defining the tomographic bins.
        **params: Additional keyword parameters forwarded to
            :func:`binny.ztomo.photoz.build_photoz_bins` (e.g., photo-z error or
            outlier model settings).

    Returns:
        Tomographic bins as returned by :func:`binny.ztomo.photoz.build_photoz_bins`.

    Raises:
        ValueError: If inputs are inconsistent or bin definitions are invalid
            (as defined by the underlying implementation).
    """
    return build_photoz_bins(z=z, nz=nz, bin_edges=bin_edges, **params)


def specz_bins(z: Any, nz: Any, bin_edges: Any, **params: Any):
    """Builds tomographic bins for a spectroscopic redshift (spec-z) sample.

    This is a public API wrapper around :func:`binny.ztomo.specz.build_specz_bins`.
    It returns a mapping of tomographic bin indices to per-bin redshift
    distributions evaluated on the input grid.

    Args:
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Bin edges defining the tomographic bins.
        **params: Additional keyword parameters forwarded to
            :func:`binny.ztomo.specz.build_specz_bins` (e.g., completeness or
            spectroscopic error model settings).

    Returns:
        Tomographic bins as returned by :func:`binny.ztomo.specz.build_specz_bins`.

    Raises:
        ValueError: If inputs are inconsistent or bin definitions are invalid
            (as defined by the underlying implementation).
    """
    return build_specz_bins(z=z, nz=nz, bin_edges=bin_edges, **params)


def tomo_bins(
    kind: str,
    z: Any,
    nz: Any,
    bin_edges: Any,
    *,
    params: Mapping[str, Any] | None = None,
):
    """Build tomographic bins for either photo-z or spec-z samples.

    This is a small dispatcher that selects :func:`photoz_bins` or
    :func:`specz_bins` based on ``kind`` and forwards parameters from ``params``.

    Args:
        kind: Binning kind selector (case-insensitive). Supported values:
            ``"photoz"``/``"photo"`` and ``"specz"``/``"spec"``.
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Bin edges defining the tomographic bins.
        params: Optional mapping of keyword parameters forwarded to the selected
            implementation.

    Returns:
        Tomographic bins as returned by :func:`photoz_bins` or :func:`specz_bins`.

    Raises:
        ValueError: If ``kind`` is not one of the supported selectors.

    Examples:
        >>> import numpy as np
        >>> from binny.api.tomo import tomo_bins
        >>> z = np.linspace(0.0, 2.0, 5)
        >>> nz = np.ones_like(z)
        >>> edges = np.array([0.0, 1.0, 2.0])
        >>> bins = tomo_bins("specz", z, nz, edges)
        >>> isinstance(bins, dict)
        True
    """
    p = dict(params or {})
    k = kind.lower()
    if k in {"photoz", "photo"}:
        return photoz_bins(z, nz, bin_edges, **p)
    if k in {"specz", "spec"}:
        return specz_bins(z, nz, bin_edges, **p)
    raise ValueError("kind must be 'photoz' or 'specz'.")
