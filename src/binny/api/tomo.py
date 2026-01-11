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


def photoz_bins(
    z: Any,
    nz: Any,
    bin_edges: Any | None = None,
    *,
    binning_scheme: Any | None = None,
    n_bins: int | None = None,
    **params: Any,
):
    """Builds tomographic bins for a photometric redshift (photo-z) sample.

    This is a public API wrapper around :func:`binny.ztomo.photoz.build_photoz_bins`.
    It returns a mapping of tomographic bin indices to per-bin redshift
    distributions evaluated on the input grid.

    You may specify bins in either of two ways:

    - Provide explicit ``bin_edges`` (recommended when you already have edges).
    - Provide ``binning_scheme`` and ``n_bins`` to construct edges internally.

    Args:
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Optional explicit bin edges defining the tomographic bins.
        binning_scheme: Optional scheme for constructing edges when
            ``bin_edges`` is ``None``.
        n_bins: Number of bins when ``binning_scheme`` is provided as a string
            (and in most simple cases).
        **params: Additional keyword parameters forwarded to
            :func:`binny.ztomo.photoz.build_photoz_bins` (e.g., photo-z error or
            outlier model settings).

    Returns:
        Tomographic bins as returned by :func:`binny.ztomo.photoz.build_photoz_bins`.

    Raises:
        ValueError: If inputs are inconsistent or bin definitions are invalid
            (as defined by the underlying implementation).
    """
    extra: dict[str, Any] = {}
    if binning_scheme is not None:
        extra["binning_scheme"] = binning_scheme
    if n_bins is not None:
        extra["n_bins"] = n_bins

    return build_photoz_bins(
        z=z,
        nz=nz,
        bin_edges=bin_edges,
        **extra,
        **params,
    )


def specz_bins(
    z: Any,
    nz: Any,
    bin_edges: Any | None = None,
    *,
    binning_scheme: Any | None = None,
    n_bins: int | None = None,
    **params: Any,
):
    """Builds tomographic bins for a spectroscopic redshift (spec-z) sample.

    This is a public API wrapper around :func:`binny.ztomo.specz.build_specz_bins`.
    It returns a mapping of tomographic bin indices to per-bin redshift
    distributions evaluated on the input grid.

    You may specify bins in either of two ways:

    - Provide explicit ``bin_edges`` (recommended for spec-z).
    - Provide ``binning_scheme`` and ``n_bins`` to construct edges internally.

    Args:
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Optional explicit bin edges defining the tomographic bins.
        binning_scheme: Optional scheme for constructing edges when
            ``bin_edges`` is ``None``.
        n_bins: Number of bins when ``binning_scheme`` is provided as a string
            (and in most simple cases).
        **params: Additional keyword parameters forwarded to
            :func:`binny.ztomo.specz.build_specz_bins` (e.g., completeness or
            spectroscopic error model settings).

    Returns:
        Tomographic bins as returned by :func:`binny.ztomo.specz.build_specz_bins`.

    Raises:
        ValueError: If inputs are inconsistent or bin definitions are invalid
            (as defined by the underlying implementation).
    """
    extra: dict[str, Any] = {}
    if binning_scheme is not None:
        extra["binning_scheme"] = binning_scheme
    if n_bins is not None:
        extra["n_bins"] = n_bins

    return build_specz_bins(
        z=z,
        nz=nz,
        bin_edges=bin_edges,
        **extra,
        **params,
    )


def tomo_bins(
    kind: str,
    z: Any,
    nz: Any,
    bin_edges: Any | None = None,
    *,
    binning_scheme: Any | None = None,
    n_bins: int | None = None,
    params: Mapping[str, Any] | None = None,
):
    """Build tomographic bins for either photo-z or spec-z samples.

    This is a small dispatcher that selects :func:`photoz_bins` or
    :func:`specz_bins` based on ``kind``.

    Bins may be specified via explicit ``bin_edges`` or via ``binning_scheme``
    and ``n_bins`` (constructed internally by the underlying implementation).

    Args:
        kind: Binning kind selector (case-insensitive). Supported values:
            ``"photoz"``/``"photo"`` and ``"specz"``/``"spec"``.
        z: Redshift grid on which the parent distribution is sampled.
        nz: Parent redshift distribution sampled on ``z``.
        bin_edges: Optional explicit bin edges defining the tomographic bins.
        binning_scheme: Optional scheme for constructing edges when
            ``bin_edges`` is ``None``.
        n_bins: Number of bins when ``binning_scheme`` is provided.
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
        >>> bins = tomo_bins("specz", z, nz, binning_scheme="equidistant", n_bins=2)
        >>> isinstance(bins, dict)
        True
    """
    p = dict(params or {})
    k = kind.lower()

    extra: dict[str, Any] = {}
    if binning_scheme is not None:
        extra["binning_scheme"] = binning_scheme
    if n_bins is not None:
        extra["n_bins"] = n_bins

    if k in {"photoz", "photo"}:
        return photoz_bins(z, nz, bin_edges, **extra, **p)

    if k in {"specz", "spec"}:
        return specz_bins(z, nz, bin_edges, **extra, **p)

    raise ValueError("kind must be 'photoz' or 'specz'.")
