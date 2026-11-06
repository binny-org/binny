"""Fit Gaussian core + tail photometric-redshift errors from catalogues."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

__all__ = ["fit_photoz_errors", "to_binny_photoz_params"]


def _fit_gaussian_mixture(
    x: np.ndarray,
    counts: np.ndarray,
    *,
    core_min_weight: float,
    n_init: int,
    seed: int,
    sigma_floor: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a zero-mean core and displaced Gaussian tail using EM."""
    rng = np.random.default_rng(seed)
    n = counts.sum()
    scale = np.sqrt(np.average(x**2, weights=counts))
    best = None

    for _ in range(n_init):
        sigma = scale * rng.uniform(0.5, 2.0, size=2)
        sigma = np.maximum(sigma, sigma_floor)
        mu = np.array([0.0, rng.choice([-1.0, 1.0]) * scale])
        frac = rng.uniform(0.0, 1.0 - core_min_weight)
        weights = np.array([1.0 - frac, frac])
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()

        previous = -np.inf

        for _ in range(max_iter):
            logp = (
                -0.5 * ((x[:, None] - mu) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2.0 * np.pi)
                + np.log(weights)
            )
            lognorm = logsumexp(logp, axis=1)
            responsibilities = np.exp(logp - lognorm[:, None]) * counts[:, None]
            nk = responsibilities.sum(axis=0)

            weights = nk / n
            weights[0] = max(weights[0], core_min_weight)
            weights[1] = 1.0 - weights[0]

            mu[1] = np.sum(responsibilities[:, 1] * x) / max(nk[1], 1e-300)

            sigma[0] = np.sqrt(np.sum(responsibilities[:, 0] * x**2) / max(nk[0], 1e-300))
            sigma[1] = np.sqrt(
                np.sum(responsibilities[:, 1] * (x - mu[1]) ** 2) / max(nk[1], 1e-300)
            )
            sigma = np.maximum(sigma, sigma_floor)

            ll = float(np.sum(counts * lognorm))
            if abs(ll - previous) <= tol * max(1.0, abs(ll)):
                break
            previous = ll

        # Evaluate likelihood using the final updated parameters.
        logp = (
            -0.5 * ((x[:, None] - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(weights)
        )
        ll = float(np.sum(counts * logsumexp(logp, axis=1)))

        if best is None or ll > best[3]:
            best = (sigma.copy(), weights.copy(), mu.copy(), ll)

    return best


def fit_photoz_errors(
    z_true: np.ndarray,
    z_obs: np.ndarray,
    *,
    hist_range: float = 0.30,
    hist_bin: float = 1e-4,
    core_min_weight: float = 0.5,
    n_init: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a Gaussian core + displaced tail to normalized photo-z residuals."""
    z_true = np.asarray(z_true, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)

    if z_true.ndim != 1 or z_obs.shape != z_true.shape:
        raise ValueError("z_true and z_obs must be matching 1D arrays.")
    if z_true.size == 0 or not np.all(np.isfinite(z_true)) or not np.all(np.isfinite(z_obs)):
        raise ValueError("Redshifts must be nonempty and finite.")
    if np.any(z_true <= -1.0):
        raise ValueError("z_true must be greater than -1.")
    if hist_range <= 0 or hist_bin <= 0 or hist_bin >= 2 * hist_range:
        raise ValueError("Invalid histogram settings.")
    if not 0.0 < core_min_weight < 1.0:
        raise ValueError("core_min_weight must lie between 0 and 1.")
    if n_init < 1:
        raise ValueError("n_init must be positive.")

    eps = (z_obs - z_true) / (1.0 + z_true)

    n_bins = max(2, round(2 * hist_range / hist_bin))
    counts, edges = np.histogram(eps, bins=n_bins, range=(-hist_range, hist_range))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if counts.sum() == 0 or np.count_nonzero(counts) < 2:
        raise ValueError("Insufficient residuals within the fitting range.")

    sigma, weights, mu, ll = _fit_gaussian_mixture(
        centers,
        counts.astype(float),
        core_min_weight=core_min_weight,
        n_init=n_init,
        seed=seed,
        sigma_floor=5.0 * hist_bin,
    )

    return {
        "sigma_c": float(sigma[0]),
        "sigma_t": float(sigma[1]),
        "m_t": float(mu[1]),
        "f_t": float(weights[1]),
        "loglike": ll,
        "n": int(eps.size),
        "n_outside": int(np.count_nonzero(np.abs(eps) > hist_range)),
    }


def to_binny_photoz_params(fit: dict) -> dict[str, float]:
    """Convert fitted core + tail parameters to Binny's photo-z model."""
    m_t = fit["m_t"]
    if m_t <= -1.0:
        raise ValueError("Tail displacement is incompatible with Binny's mean scale.")

    return {
        "mean_scale": 1.0,
        "mean_offset": 0.0,
        "scatter_scale": fit["sigma_c"],
        "outlier_frac": fit["f_t"],
        "outlier_mean_scale": 1.0 + m_t,
        "outlier_mean_offset": -m_t,
        "outlier_scatter_scale": fit["sigma_t"],
    }
