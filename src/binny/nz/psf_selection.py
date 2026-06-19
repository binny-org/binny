"""Calibration tools for PSF-selected galaxy redshift distributions.

This module provides utilities for modelling how survey depth and image
quality change the effective source sample used in weak-lensing forecasts.

The main goal is to derive empirical redshift distributions and galaxy
number densities from mock or simulated catalogs after applying both:

- a magnitude-limit selection;
- a PSF-dependent shear-selection weight.

These routines are intended for forecasting studies where changes in the PSF
or limiting magnitude need to be propagated into source redshift distributions,
tomographic samples, and effective galaxy densities.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from binny.surveys.sky import deg2_to_arcmin2

__all__ = [
    "resolution_factor",
    "shear_selection_weight",
    "weighted_nz_from_mock",
    "effective_number_density_from_mock",
    "calibrate_psf_depth_from_mock",
]


SelectionKind = Literal["hard", "sigmoid"]


def resolution_factor(
    r_gal: np.ndarray,
    r_psf: float,
) -> np.ndarray:
    """
    Compute the resolution factor of galaxies relative to the PSF.

    The resolution factor describes how well a galaxy is resolved compared
    with the point-spread function. Values close to zero correspond to
    poorly resolved galaxies, while values close to one correspond to
    well-resolved galaxies.

    Args:
        r_gal: Intrinsic or pre-seeing galaxy sizes.
        r_psf: Characteristic PSF size in the same units as ``r_gal``.

    Returns:
        Galaxy resolution factors.

    Raises:
        ValueError: If ``r_psf`` is negative.
    """
    if r_psf < 0:
        raise ValueError("r_psf must be >= 0")

    r_gal = np.asarray(r_gal, dtype=float)
    return r_gal**2 / (r_gal**2 + float(r_psf) ** 2)


def shear_selection_weight(
    r_gal: np.ndarray,
    *,
    r_psf: float,
    r_min: float = 0.3,
    kind: SelectionKind = "sigmoid",
    width: float = 0.05,
) -> np.ndarray:
    """
    Compute PSF-dependent shear-selection weights.

    This function assigns each galaxy a weight describing whether it remains
    usable for shear measurements after accounting for PSF resolution. The
    selection can be either a hard resolution threshold or a smooth sigmoid
    transition around the threshold.

    Args:
        r_gal: Intrinsic or pre-seeing galaxy sizes.
        r_psf: Characteristic PSF size in the same units as ``r_gal``.
        r_min: Minimum resolution factor required for shear selection.
        kind: Selection model used to convert resolution into weights.
        width: Width of the sigmoid transition when ``kind="sigmoid"``.

    Returns:
        Shear-selection weights between zero and one.

    Raises:
        ValueError: If ``r_min`` is outside ``[0, 1]``.
        ValueError: If ``kind`` is not supported.
        ValueError: If ``width`` is not positive for sigmoid selection.
    """
    if not 0 <= r_min <= 1:
        raise ValueError("r_min must be between 0 and 1")

    resolution = resolution_factor(r_gal, r_psf)

    if kind == "hard":
        return (resolution >= float(r_min)).astype(float)

    if kind == "sigmoid":
        if width <= 0:
            raise ValueError("width must be > 0 for sigmoid selection")
        x = (resolution - float(r_min)) / float(width)
        return 1.0 / (1.0 + np.exp(-x))

    raise ValueError("kind must be 'hard' or 'sigmoid'")


def weighted_nz_from_mock(
    z_true: np.ndarray,
    mag: np.ndarray,
    r_gal: np.ndarray,
    *,
    maglim: float,
    r_psf: float,
    z_edges: np.ndarray,
    r_min: float = 0.3,
    selection_kind: SelectionKind = "sigmoid",
    width: float = 0.05,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a PSF-selected redshift distribution from a mock catalog.

    This function estimates the source redshift distribution after applying
    a magnitude limit and a PSF-dependent shear-selection weight. It returns
    the redshift-bin centers and the corresponding weighted distribution.

    Args:
        z_true: True galaxy redshifts from the mock catalog.
        mag: Apparent magnitudes of the same galaxies.
        r_gal: Galaxy sizes of the same galaxies.
        maglim: Limiting magnitude used to define the source sample.
        r_psf: Characteristic PSF size in the same units as ``r_gal``.
        z_edges: Redshift-bin edges used to histogram the selected sample.
        r_min: Minimum resolution factor required for shear selection.
        selection_kind: Selection model used to convert resolution into weights.
        width: Width of the sigmoid transition for smooth selection.
        normalize: Whether to normalize the redshift distribution to unit area.

    Returns:
        Tuple containing redshift-bin centers and the weighted redshift distribution.

    Raises:
        ValueError: If input catalog arrays do not have matching shapes.
        ValueError: If ``z_edges`` does not contain at least two bin edges.
    """
    z_true = np.asarray(z_true, dtype=float)
    mag = np.asarray(mag, dtype=float)
    r_gal = np.asarray(r_gal, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)

    if z_true.shape != mag.shape or z_true.shape != r_gal.shape:
        raise ValueError("z_true, mag, and r_gal must have matching shapes")
    if z_edges.ndim != 1 or z_edges.size < 2:
        raise ValueError("z_edges must be a one-dimensional array with at least two edges")

    valid = np.isfinite(z_true) & np.isfinite(mag) & np.isfinite(r_gal)
    selected = valid & (z_true >= 0) & (mag <= float(maglim))

    weights = shear_selection_weight(
        r_gal[selected],
        r_psf=r_psf,
        r_min=r_min,
        kind=selection_kind,
        width=width,
    )

    nz, _ = np.histogram(
        z_true[selected],
        bins=z_edges,
        weights=weights,
    )

    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    nz = nz.astype(float)

    if normalize:
        area = np.trapezoid(nz, z_mid)
        if area > 0:
            nz = nz / area

    return z_mid, nz


def effective_number_density_from_mock(
    mag: np.ndarray,
    r_gal: np.ndarray,
    *,
    maglim: float,
    r_psf: float,
    area_deg2: float,
    r_min: float = 0.3,
    selection_kind: SelectionKind = "hard",
    width: float = 0.05,
) -> float:
    """
    Estimate the PSF-selected effective galaxy density from a mock catalog.

    This function computes the weighted source density after applying a
    magnitude limit and a PSF-dependent shear-selection weight.

    Args:
        mag: Apparent magnitudes from the mock catalog.
        r_gal: Galaxy sizes of the same galaxies.
        maglim: Limiting magnitude used to define the source sample.
        r_psf: Characteristic PSF size in the same units as ``r_gal``.
        area_deg2: Survey or mock area in square degrees.
        r_min: Minimum resolution factor required for shear selection.
        selection_kind: Selection model used to convert resolution into weights.
        width: Width of the sigmoid transition for smooth selection.

    Returns:
        Weighted galaxy number density in galaxies per arcmin².

    Raises:
        ValueError: If input catalog arrays do not have matching shapes.
        ValueError: If ``area_deg2`` is not positive.
    """
    mag = np.asarray(mag, dtype=float)
    r_gal = np.asarray(r_gal, dtype=float)

    if mag.shape != r_gal.shape:
        raise ValueError("mag and r_gal must have matching shapes")
    if area_deg2 <= 0:
        raise ValueError("area_deg2 must be > 0")

    valid = np.isfinite(mag) & np.isfinite(r_gal)
    selected = valid & (mag <= float(maglim))

    weights = shear_selection_weight(
        r_gal[selected],
        r_psf=r_psf,
        r_min=r_min,
        kind=selection_kind,
        width=width,
    )

    area_arcmin2 = deg2_to_arcmin2(area_deg2)
    return float(np.sum(weights) / area_arcmin2)


def calibrate_psf_depth_from_mock(
    z_true: np.ndarray,
    mag: np.ndarray,
    r_gal: np.ndarray,
    *,
    maglims: np.ndarray,
    r_psf_values: np.ndarray,
    area_deg2: float,
    z_edges: np.ndarray,
    r_min: float = 0.3,
    selection_kind: SelectionKind = "sigmoid",
    width: float = 0.05,
    normalize_nz: bool = True,
) -> dict[str, Any]:
    """
    Calibrate source redshift distributions versus survey depth and PSF size.

    This routine evaluates how the source redshift distribution and effective
    galaxy number density change across a grid of limiting magnitudes and PSF
    sizes. It is intended for forecasting studies that compare baseline and
    as-built survey scenarios.

    Args:
        z_true: True galaxy redshifts from the mock catalog.
        mag: Apparent magnitudes of the same galaxies.
        r_gal: Galaxy sizes of the same galaxies.
        maglims: Limiting magnitudes to evaluate.
        r_psf_values: PSF sizes to evaluate.
        area_deg2: Survey or mock area in square degrees.
        z_edges: Redshift-bin edges used to histogram the selected samples.
        r_min: Minimum resolution factor required for shear selection.
        selection_kind: Selection model used to convert resolution into weights.
        width: Width of the sigmoid transition for smooth selection.
        normalize_nz: Whether to normalize each redshift distribution to unit area.

    Returns:
        Dictionary containing the calibration grid, redshift distributions,
        and effective galaxy densities.

    Raises:
        ValueError: If input catalog arrays do not have matching shapes.
        ValueError: If ``area_deg2`` is not positive.
    """
    z_true = np.asarray(z_true, dtype=float)
    mag = np.asarray(mag, dtype=float)
    r_gal = np.asarray(r_gal, dtype=float)
    maglims = np.asarray(maglims, dtype=float)
    r_psf_values = np.asarray(r_psf_values, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)

    if z_true.shape != mag.shape or z_true.shape != r_gal.shape:
        raise ValueError("z_true, mag, and r_gal must have matching shapes")
    if area_deg2 <= 0:
        raise ValueError("area_deg2 must be > 0")

    results: list[dict[str, Any]] = []

    for maglim in maglims:
        for r_psf in r_psf_values:
            z_mid, nz = weighted_nz_from_mock(
                z_true=z_true,
                mag=mag,
                r_gal=r_gal,
                maglim=float(maglim),
                r_psf=float(r_psf),
                z_edges=z_edges,
                r_min=r_min,
                selection_kind=selection_kind,
                width=width,
                normalize=normalize_nz,
            )

            neff = effective_number_density_from_mock(
                mag=mag,
                r_gal=r_gal,
                maglim=float(maglim),
                r_psf=float(r_psf),
                area_deg2=area_deg2,
                r_min=r_min,
                selection_kind=selection_kind,
                width=width,
            )

            results.append(
                {
                    "maglim": float(maglim),
                    "r_psf": float(r_psf),
                    "z": z_mid,
                    "nz": nz,
                    "neff_arcmin2": neff,
                }
            )

    return {
        "ok": True,
        "model": "psf_depth_selection_from_mock",
        "selection": {
            "kind": selection_kind,
            "r_min": float(r_min),
            "width": float(width),
        },
        "area_deg2": float(area_deg2),
        "z_edges": z_edges,
        "maglims": maglims,
        "r_psf_values": r_psf_values,
        "results": results,
    }
