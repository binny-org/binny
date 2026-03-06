"""Calibration tools for magnitude-limited galaxy redshift distributions.

This module provides utilities for calibrating the parameters of the
Smail redshift distribution and related survey scaling relations from
mock or simulated galaxy catalogs.

The main goal is to derive empirical mappings between survey depth
(limiting magnitude) and the statistical properties of a galaxy sample,
including:

- the parameters of the Smail redshift distribution
- the evolution of the redshift scale parameter with magnitude limit
- the number density of galaxies as a function of survey depth

These calibration routines allow realistic survey specifications to be
derived directly from simulated catalogs rather than adopting fixed
parametric assumptions. The resulting fits can then be used to generate
analytic redshift distributions and galaxy densities for forecasting
and survey design studies.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.special import gammaln

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from binny.surveys.sky import deg2_to_arcmin2

__all__ = [
    "fit_smail_params_from_mock",
    "fit_z0_of_maglim_from_mock",
    "eval_z0_of_maglim",
    "fit_ngal_of_maglim_from_mock",
    "eval_ngal_of_maglim",
    "calibrate_depth_smail_from_mock",
]


def _smail_logpdf(z: np.ndarray, z0: float, alpha: float, beta: float) -> np.ndarray:
    """
    Evaluate the log-probability density of the normalized Smail distribution.

    The Smail distribution is a commonly used parametric model for galaxy
    redshift distributions in magnitude-limited surveys. It describes the
    shape of the population redshift distribution using three parameters:
    a shape parameter (alpha), a tail parameter (beta), and a scale
    parameter (z0).

    This function returns the logarithm of the probability density at the
    provided redshift values. Using the log-density improves numerical
    stability when fitting the distribution to large samples.

    Args:
        z (np.ndarray):
            Redshift values where the probability density should be evaluated.
        z0 (float):
            Characteristic redshift scale of the distribution.
        alpha (float):
            Low-redshift shape parameter controlling the rise of the distribution.
        beta (float):
            High-redshift tail parameter controlling the falloff of the distribution.

    Returns:
        np.ndarray:
            Logarithm of the probability density evaluated at each redshift.
    """
    z = np.asarray(z, dtype=float)
    if z0 <= 0 or beta <= 0 or alpha <= -1:
        return np.full_like(z, -np.inf)

    zz = np.maximum(z, 1e-300)
    t = (alpha + 1.0) / beta

    logC = np.log(beta) - np.log(z0) - gammaln(t)
    log_pow = alpha * (np.log(zz) - np.log(z0))
    expo = -((zz / z0) ** beta)

    out = logC + log_pow + expo
    out = np.where(z >= 0, out, -np.inf)
    return out


def fit_smail_params_from_mock(
    z_samples: np.ndarray,
    *,
    alpha_bounds: tuple[float, float] = (0.2, 8.0),
    beta_bounds: tuple[float, float] = (0.2, 6.0),
    z0_bounds: tuple[float, float] = (0.01, 5.0),
    x0: tuple[float, float, float] | None = None,  # (alpha, beta, z0)
    z_max: float | None = None,
    min_n: int = 200,
) -> dict[str, Any]:
    """
    Infer Smail distribution parameters from a mock redshift sample.

    This function estimates the parameters of the Smail redshift distribution
    (alpha, beta, and z0) from a set of galaxy redshifts. The fitted model
    represents the underlying redshift distribution of the galaxy sample.

    The routine is typically applied to simulated or mock catalogs in order
    to calibrate the parameters of analytic redshift distributions used in
    forecasting or survey modeling.

    Args:
        z_samples (np.ndarray):
            Array of galaxy redshifts drawn from a mock or simulated catalog.
        alpha_bounds (tuple[float, float], optional):
            Allowed range for the alpha parameter.
        beta_bounds (tuple[float, float], optional):
            Allowed range for the beta parameter.
        z0_bounds (tuple[float, float], optional):
            Allowed range for the z0 parameter.
        x0 (tuple[float, float, float] | None, optional):
            Initial parameter guess for the fit.
        z_max (float | None, optional):
            Maximum redshift used when fitting the distribution.
        min_n (int, optional):
            Minimum number of redshift samples required to perform the fit.

    Returns:
        dict[str, Any]:
            Dictionary containing the fitted parameters, fit status, and
            summary information about the calibration sample.
    """
    if minimize is None:
        raise RuntimeError("scipy.optimize.minimize is required for Smail MLE fitting.")

    z = np.asarray(z_samples, dtype=float)
    z = z[np.isfinite(z)]
    z = z[z >= 0]
    if z_max is not None:
        z = z[z <= float(z_max)]

    if z.size < min_n:
        return {"ok": False, "reason": "too_few_samples", "params": None, "n": int(z.size)}

    if x0 is None:
        alpha0 = 2.0
        beta0 = 1.5
        # crude but good initializer
        z0_0 = max(z0_bounds[0], np.median(z) / 1.4)
        x0 = (alpha0, beta0, z0_0)

    bounds = [alpha_bounds, beta_bounds, z0_bounds]

    def nll(x: np.ndarray) -> float:
        a, b, z0 = float(x[0]), float(x[1]), float(x[2])
        lp = _smail_logpdf(z, z0=z0, alpha=a, beta=b)
        if not np.all(np.isfinite(lp)):
            return np.inf
        return float(-np.sum(lp))

    res = minimize(
        nll,
        x0=np.array(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        return {
            "ok": False,
            "reason": "optimizer_failed",
            "message": str(res.message),
            "params": None,
            "n": int(z.size),
        }

    a, b, z0 = map(float, res.x)
    return {
        "ok": True,
        "method": "mle_l_bfgs_b",
        "params": {"alpha": a, "beta": b, "z0": z0},
        "n": int(z.size),
        "fun": float(res.fun),
    }


def fit_z0_of_maglim_from_mock(
    z_true: np.ndarray,
    mag: np.ndarray,
    *,
    maglims: np.ndarray,
    alpha: float,
    beta: float,
    z0_law: Literal["linear", "poly2"] = "linear",
    z0_bounds: tuple[float, float] = (0.01, 5.0),
    z_max: float | None = None,
    min_n_per_cut: int = 200,
) -> dict[str, Any]:
    """
    Calibrate the relation between survey depth and the Smail scale parameter.

    This function measures how the characteristic redshift scale (z0) of a
    galaxy population evolves with survey limiting magnitude. For each
    magnitude limit, galaxies are selected from a mock catalog and the
    corresponding redshift distribution is fitted with a Smail model with
    fixed shape parameters.

    The resulting z0 measurements are then used to fit a smooth empirical
    relation between the magnitude limit and the redshift scale.

    Args:
        z_true (np.ndarray):
            True redshifts of galaxies in the mock catalog.
        mag (np.ndarray):
            Apparent magnitudes of galaxies in the same catalog.
        maglims (np.ndarray):
            Limiting magnitudes at which the calibration should be evaluated.
        alpha (float):
            Fixed alpha parameter of the Smail distribution.
        beta (float):
            Fixed beta parameter of the Smail distribution.
        z0_law (Literal["linear", "poly2"], optional):
            Functional form used to model the z0–magnitude relation.
        z0_bounds (tuple[float, float], optional):
            Allowed range for the fitted z0 values.
        z_max (float | None, optional):
            Maximum redshift considered when fitting the distribution.
        min_n_per_cut (int, optional):
            Minimum number of galaxies required for a magnitude selection
            to be included in the calibration.

    Returns:
        dict[str, Any]:
            Dictionary containing the measured z0 values at each magnitude
            limit and the parameters of the fitted z0–magnitude relation.
    """
    if minimize is None:
        raise RuntimeError("scipy.optimize.minimize is required for z0(maglim) fitting.")

    z_true = np.asarray(z_true, dtype=float)
    mag = np.asarray(mag, dtype=float)
    maglims = np.asarray(maglims, dtype=float)

    z0_pts = np.full_like(maglims, np.nan, dtype=float)
    nsel_pts = np.zeros_like(maglims, dtype=int)

    # 1D MLE for z0 with fixed alpha,beta
    def fit_z0_only(zs: np.ndarray) -> float:
        z = zs[np.isfinite(zs)]
        z = z[z >= 0]
        if z_max is not None:
            z = z[z <= float(z_max)]
        if z.size < min_n_per_cut:
            return np.nan

        def nll_z0(x: np.ndarray) -> float:
            z0 = float(x[0])
            lp = _smail_logpdf(z, z0=z0, alpha=float(alpha), beta=float(beta))
            if not np.all(np.isfinite(lp)):
                return np.inf
            return float(-np.sum(lp))

        x0 = np.array([max(z0_bounds[0], np.median(z) / 1.4)], dtype=float)
        res = minimize(nll_z0, x0=x0, method="L-BFGS-B", bounds=[z0_bounds])
        return float(res.x[0]) if res.success else np.nan

    for i, mlim in enumerate(maglims):
        sel = np.isfinite(z_true) & np.isfinite(mag) & (mag <= mlim) & (z_true >= 0)
        z_sel = z_true[sel]
        nsel_pts[i] = int(z_sel.size)
        z0_pts[i] = fit_z0_only(z_sel)

    m = np.isfinite(z0_pts)
    if np.sum(m) < 2:
        return {"ok": False, "reason": "too_few_valid_points", "points": None, "fit": None}

    if z0_law == "linear":
        a, b = np.polyfit(maglims[m], z0_pts[m], deg=1)  # z0 = a*m + b
        fit = {"law": "linear", "a": float(a), "b": float(b)}
    elif z0_law == "poly2":
        c2, c1, c0 = np.polyfit(maglims[m], z0_pts[m], deg=2)
        fit = {"law": "poly2", "c2": float(c2), "c1": float(c1), "c0": float(c0)}
    else:
        raise ValueError("z0_law must be 'linear' or 'poly2'")

    return {
        "ok": True,
        "points": {"maglim": maglims, "z0": z0_pts, "n_selected": nsel_pts},
        "fit": fit,
        "fixed_smail": {"alpha": float(alpha), "beta": float(beta)},
        "z0_bounds": tuple(map(float, z0_bounds)),
    }


def eval_z0_of_maglim(maglim: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """
    Evaluate the calibrated z0–magnitude relation.

    This function computes the characteristic Smail redshift scale for one
    or more magnitude limits using a previously calibrated z0–magnitude
    relation.

    Args:
        maglim (np.ndarray):
            Limiting magnitudes at which the relation should be evaluated.
        fit (dict[str, Any]):
            Calibration result describing the fitted z0–magnitude relation.

    Returns:
        np.ndarray:
            Predicted z0 values corresponding to the input magnitude limits.
    """
    m = np.asarray(maglim, dtype=float)
    if fit["law"] == "linear":
        return fit["a"] * m + fit["b"]
    if fit["law"] == "poly2":
        return fit["c2"] * m**2 + fit["c1"] * m + fit["c0"]
    raise ValueError(f"Unknown law: {fit['law']}")


def fit_ngal_of_maglim_from_mock(
    mag: np.ndarray,
    *,
    maglims: np.ndarray,
    area_deg2: float,
    ngal_law: Literal["linear", "loglinear"] = "loglinear",
) -> dict[str, Any]:
    """
    Calibrate the galaxy number density as a function of survey depth.

    This function measures the number density of galaxies in a mock catalog
    for a series of magnitude limits and fits a smooth empirical relation
    between survey depth and galaxy density.

    The resulting relation can be used to predict the effective galaxy
    density expected for different magnitude limits in survey forecasts.

    Args:
        mag (np.ndarray):
            Apparent magnitudes of galaxies in the mock catalog.
        maglims (np.ndarray):
            Limiting magnitudes used to define magnitude-limited samples.
        area_deg2 (float):
            Survey area of the mock catalog in square degrees.
        ngal_law (Literal["linear", "loglinear"], optional):
            Functional form used to model the density–magnitude relation.

    Returns:
        dict[str, Any]:
            Dictionary containing measured galaxy densities at each magnitude
            limit and the parameters of the fitted density relation.
    """
    mag = np.asarray(mag, dtype=float)
    mag = mag[np.isfinite(mag)]
    maglims = np.asarray(maglims, dtype=float)

    if area_deg2 <= 0:
        raise ValueError("area_deg2 must be > 0")
    area_arcmin2 = deg2_to_arcmin2(area_deg2)

    ngal_pts = np.zeros_like(maglims, dtype=float)
    nsel_pts = np.zeros_like(maglims, dtype=int)
    for i, mlim in enumerate(maglims):
        nsel = int(np.sum(mag <= mlim))
        nsel_pts[i] = nsel
        ngal_pts[i] = nsel / area_arcmin2

    g = ngal_pts > 0
    if np.sum(g) < 2:
        return {"ok": False, "reason": "too_few_valid_points", "points": None, "fit": None}

    if ngal_law == "linear":
        p, q = np.polyfit(maglims[g], ngal_pts[g], deg=1)  # ngal = p*m + q
        fit = {"law": "linear", "p": float(p), "q": float(q)}
    elif ngal_law == "loglinear":
        s, t = np.polyfit(maglims[g], np.log10(ngal_pts[g]), deg=1)  # log10(ngal)=s*m+t
        fit = {"law": "loglinear", "s": float(s), "t": float(t)}
    else:
        raise ValueError("ngal_law must be 'linear' or 'loglinear'")

    return {
        "ok": True,
        "points": {"maglim": maglims, "ngal_arcmin2": ngal_pts, "n_selected": nsel_pts},
        "fit": fit,
        "area_deg2": float(area_deg2),
    }


def eval_ngal_of_maglim(maglim: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """
    Evaluate the calibrated galaxy density–magnitude relation.

    This function computes the predicted galaxy number density for a set of
    magnitude limits using a previously calibrated density relation.

    Args:
        maglim (np.ndarray):
            Limiting magnitudes at which the density should be evaluated.
        fit (dict[str, Any]):
            Calibration result describing the fitted galaxy density relation.

    Returns:
        np.ndarray:
            Predicted galaxy number densities in galaxies per arcmin².
    """
    m = np.asarray(maglim, dtype=float)
    if fit["law"] == "linear":
        return fit["p"] * m + fit["q"]
    if fit["law"] == "loglinear":
        return 10.0 ** (fit["s"] * m + fit["t"])
    raise ValueError(f"Unknown law: {fit['law']}")


def calibrate_depth_smail_from_mock(
    z_true: np.ndarray,
    mag: np.ndarray,
    *,
    maglims: np.ndarray,
    area_deg2: float,
    infer_alpha_beta_from: Literal["deep_cut", "all_selected_at_maglim"] = "deep_cut",
    alpha_beta_maglim: float | None = None,  # used if infer_alpha_beta_from="deep_cut"
    z_max: float | None = None,
) -> dict[str, Any]:
    """
    Run an end-to-end calibration of survey depth scaling relations.

    This routine performs a complete calibration of the relations linking
    survey limiting magnitude to both the redshift distribution and the
    galaxy number density of a sample.

    The procedure estimates the shape parameters of the Smail redshift
    distribution from a representative galaxy sample, calibrates how the
    redshift scale parameter varies with magnitude limit, and measures the
    corresponding galaxy number density.

    The resulting calibration can be used to construct realistic analytic
    redshift distributions and galaxy densities for survey forecasts.

    Args:
        z_true (np.ndarray):
            True redshifts of galaxies in the mock catalog.
        mag (np.ndarray):
            Apparent magnitudes of the same galaxies.
        maglims (np.ndarray):
            Limiting magnitudes defining magnitude-limited samples.
        area_deg2 (float):
            Survey area of the mock catalog in square degrees.
        infer_alpha_beta_from (Literal["deep_cut", "all_selected_at_maglim"], optional):
            Strategy used to determine the shape parameters of the Smail
            distribution.
        alpha_beta_maglim (float | None, optional):
            Magnitude limit defining the deep sample used to infer the
            Smail shape parameters.
        z_max (float | None, optional):
            Maximum redshift included when fitting redshift distributions.

    Returns:
        dict[str, Any]:
            Dictionary containing the calibrated Smail parameters,
            the fitted z0–magnitude relation, and the galaxy density
            calibration.
    """
    z_true = np.asarray(z_true, dtype=float)
    mag = np.asarray(mag, dtype=float)
    maglims = np.asarray(maglims, dtype=float)

    if infer_alpha_beta_from == "deep_cut":
        if alpha_beta_maglim is None:
            alpha_beta_maglim = float(np.nanmax(maglims))
        sel = (
            np.isfinite(z_true)
            & np.isfinite(mag)
            & (mag <= float(alpha_beta_maglim))
            & (z_true >= 0)
        )
        z_for_ab = z_true[sel]
    else:
        # all galaxies (no mag cut) — only makes sense if mag array already
        # represents my sample selection
        sel = np.isfinite(z_true) & (z_true >= 0)
        z_for_ab = z_true[sel]

    ab = fit_smail_params_from_mock(z_for_ab, z_max=z_max)
    if not ab["ok"]:
        return {"ok": False, "reason": "alpha_beta_fit_failed", "alpha_beta_fit": ab}

    alpha = float(ab["params"]["alpha"])
    beta = float(ab["params"]["beta"])

    z0_cal = fit_z0_of_maglim_from_mock(
        z_true=z_true,
        mag=mag,
        maglims=maglims,
        alpha=alpha,
        beta=beta,
        z_max=z_max,
    )
    ngal_cal = fit_ngal_of_maglim_from_mock(
        mag=mag,
        maglims=maglims,
        area_deg2=area_deg2,
        ngal_law="loglinear",
    )

    return {
        "ok": bool(z0_cal.get("ok", False) and ngal_cal.get("ok", False)),
        "alpha_beta_fit": ab,
        "z0_of_maglim": z0_cal,
        "ngal_of_maglim": ngal_cal,
    }
