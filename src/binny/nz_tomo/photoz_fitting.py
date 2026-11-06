"""Fit Gaussian core and tail photometric-redshift errors from catalogues.

This module fits an unbinned two-component Gaussian mixture to photometric
redshift residuals using the expectation-maximization algorithm.

The residual is defined as

    epsilon = (z_obs - z_true) / (1 + z_true).

The mixture consists of a zero-mean Gaussian core and a displaced Gaussian
tail. Multiple random initializations are used, and the solution with the
highest log-likelihood is retained.

The fitted parameters use Binny's photometric-redshift naming conventions.
The fitted tail displacement can be converted to Binny's mean scale and
offset using ``to_binny_photoz_params``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

__all__ = ["fit_photoz_errors", "to_binny_photoz_params"]


def _log_probability(
    x: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate the log probability density of a Gaussian mixture.

    Computes the mixture log density using the log-sum-exp operation
    to avoid numerical underflow when component probabilities are small.

    Args:
        x: One-dimensional array of residual values.
        sigma: Standard deviations of the Gaussian components, corresponding
            to ``scatter_scale`` and ``outlier_scatter_scale``.
        mu: Means of the Gaussian components. The core mean is zero, and
            the tail mean corresponds to ``outlier_mean_shift``.
        weights: Mixture weights of the Gaussian components, corresponding
            to the core fraction and ``outlier_frac``.

    Returns:
        Array of mixture log probability densities, with one value
        per input residual.
    """
    logp = (
        -0.5 * ((x[:, None] - mu) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        + np.log(weights)
    )
    return logsumexp(logp, axis=1)


def _fit_gaussian_mixture(
    x: np.ndarray,
    *,
    core_min_weight: float,
    n_init: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, int]:
    """Fit a Gaussian core and displaced tail using expectation-maximization.

    Fits a two-component Gaussian mixture in which the core mean is fixed
    at zero and the tail mean is free. Both component standard deviations
    and the mixture weights are estimated from the unbinned residuals.

    Each initialization alternates between computing component
    responsibilities and updating the mixture parameters. The core
    mixture weight is constrained to be at least ``core_min_weight``.

    The fit with the highest final log-likelihood is selected across
    all random initializations.

    Args:
        x: One-dimensional array of finite, non-degenerate residuals.
        core_min_weight: Minimum allowed mixture weight of the
            zero-mean core component.
        n_init: Number of random expectation-maximization initializations.
        seed: Random seed used to initialize the mixture parameters.
        max_iter: Maximum number of expectation-maximization iterations
            per initialization.
        tol: Relative tolerance for changes in the total
            log-likelihood between consecutive iterations.

    Returns:
        Tuple containing:

            - sigma: Standard deviations corresponding to
              ``scatter_scale`` and ``outlier_scatter_scale``.
            - weights: Core and tail mixture weights, with the
              tail weight corresponding to ``outlier_frac``.
            - mu: Component means, with the core mean fixed at zero
              and the tail mean corresponding to ``outlier_mean_shift``.
            - ll: Final total log-likelihood of the selected fit.
            - converged: Whether the selected initialization satisfied
              the convergence criterion.
            - n_iter: Number of expectation-maximization iterations
              performed by the selected initialization.

    Notes:
        Component standard deviations are bounded below by a small
        scale-dependent floor to avoid singular Gaussian components.

        The convergence criterion uses the relative change in the
        total log-likelihood. Convergence does not guarantee that the
        global maximum of the mixture likelihood has been found.
    """
    rng = np.random.default_rng(seed)

    scale = np.sqrt(np.mean(x**2))
    sigma_floor = max(1e-6, scale * 1e-4)

    best = None

    for _ in range(n_init):
        sigma = np.maximum(
            scale * rng.uniform(0.5, 2.0, size=2),
            sigma_floor,
        )
        mu = np.array([0.0, rng.choice([-1.0, 1.0]) * scale])

        frac = rng.uniform(1e-3, 1.0 - core_min_weight)
        weights = np.array([1.0 - frac, frac])

        previous = -np.inf
        converged = False

        for _iteration in range(1, max_iter + 1):
            logp = (
                -0.5 * ((x[:, None] - mu) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2.0 * np.pi)
                + np.log(weights)
            )
            lognorm = logsumexp(logp, axis=1)
            responsibilities = np.exp(logp - lognorm[:, None])

            nk = responsibilities.sum(axis=0)

            weights[0] = max(nk[0] / x.size, core_min_weight)
            weights[1] = 1.0 - weights[0]

            mu[1] = np.sum(responsibilities[:, 1] * x) / max(nk[1], 1e-300)

            sigma[0] = np.sqrt(np.sum(responsibilities[:, 0] * x**2) / max(nk[0], 1e-300))
            sigma[1] = np.sqrt(
                np.sum(responsibilities[:, 1] * (x - mu[1]) ** 2) / max(nk[1], 1e-300)
            )
            sigma = np.maximum(sigma, sigma_floor)

            ll = float(np.sum(_log_probability(x, sigma, mu, weights)))

            if np.isfinite(previous) and abs(ll - previous) <= tol * max(1.0, abs(ll)):
                converged = True
                break

            previous = ll

        ll = float(np.sum(_log_probability(x, sigma, mu, weights)))

        if best is None or ll > best[3]:
            best = (
                sigma.copy(),
                weights.copy(),
                mu.copy(),
                ll,
                converged,
                _iteration,
            )

    return best


def _ks_statistic(
    x: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic for the fitted mixture.

    Compares the empirical cumulative distribution of the residuals
    with the cumulative distribution of the fitted Gaussian mixture.

    The statistic is the maximum absolute difference between the
    empirical and model cumulative distribution functions.

    Args:
        x: One-dimensional array of residual values.
        sigma: Standard deviations of the Gaussian components, corresponding
            to ``scatter_scale`` and ``outlier_scatter_scale``.
        mu: Means of the Gaussian components, with the tail mean
            corresponding to ``outlier_mean_shift``.
        weights: Mixture weights of the Gaussian components, with the
            tail weight corresponding to ``outlier_frac``.

    Returns:
        Kolmogorov-Smirnov statistic between zero and one.

    Notes:
        The statistic is evaluated using parameters estimated from the
        same catalogue. Standard fixed-distribution Kolmogorov-Smirnov
        significance probabilities are therefore not directly applicable.
    """
    sorted_x = np.sort(x)
    model_cdf = np.sum(
        weights * ndtr((sorted_x[:, None] - mu) / sigma),
        axis=1,
    )

    n = sorted_x.size
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(n) / n

    return float(
        max(
            np.max(empirical_upper - model_cdf),
            np.max(model_cdf - empirical_lower),
        )
    )


def fit_photoz_errors(
    z_true: np.ndarray,
    z_obs: np.ndarray,
    *,
    core_min_weight: float = 0.5,
    n_init: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-8,
    seed: int = 0,
    return_metadata: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float | int | bool]]:
    """Fit a Gaussian core and displaced tail to photometric-redshift residuals.

    Fits an unbinned two-component Gaussian mixture to the normalized
    photometric-redshift residual

        epsilon = (z_obs - z_true) / (1 + z_true).

    The probability density is

        p(epsilon) = (1 - outlier_frac)
                     * N(epsilon; 0, scatter_scale**2)
                   + outlier_frac
                     * N(epsilon; outlier_mean_shift,
                         outlier_scatter_scale**2),

    where ``scatter_scale`` is the core scatter,
    ``outlier_scatter_scale`` is the tail scatter,
    ``outlier_mean_shift`` is the tail displacement, and
    ``outlier_frac`` is the tail mixture fraction.

    The core mean is fixed at zero, while the tail mean, both
    standard deviations, and the mixture fraction are fitted.

    The fit uses expectation-maximization with multiple random
    initializations. All finite catalogue residuals are included;
    no histogram binning or residual-range truncation is applied.

    Args:
        z_true: One-dimensional array of true galaxy redshifts.
        z_obs: One-dimensional array of observed photometric redshifts,
            with the same shape as ``z_true``.
        core_min_weight: Minimum allowed core mixture fraction.
            Must lie strictly between zero and one. The default
            value of 0.5 restricts ``outlier_frac`` to at most 0.5.
        n_init: Number of random expectation-maximization initializations.
            Must be positive. Defaults to 10.
        max_iter: Maximum number of expectation-maximization iterations
            per initialization. Must be positive. Defaults to 1000.
        tol: Relative log-likelihood convergence tolerance.
            Must be positive and finite. Defaults to 1e-8.
        seed: Random seed controlling the expectation-maximization
            initializations. Defaults to 0.

    Returns:
        Dictionary containing the fitted photometric-redshift parameters:

            - scatter_scale: Standard deviation of the zero-mean core.
            - outlier_scatter_scale: Standard deviation of the Gaussian tail.
            - outlier_mean_shift: Mean displacement of the Gaussian tail.
            - outlier_frac: Fitted tail mixture fraction.

        If ``return_metadata=True``, returns a tuple containing the
        fitted parameter dictionary and a metadata dictionary with:

            - loglike: Total log-likelihood of the selected fit.
            - n: Number of catalogue objects used in the fit.
            - n_outside: Number of excluded residuals, always zero.
            - converged: Whether the selected optimization converged.
            - n_iter: Number of iterations in the selected optimization.
            - aic: Akaike information criterion.
            - bic: Bayesian information criterion.
            - ks_statistic: Kolmogorov-Smirnov statistic.

    Raises:
        ValueError: If the catalogue arrays have incompatible shapes,
            contain insufficient or non-finite values, or contain
            true redshifts less than or equal to -1.
        ValueError: If the residual distribution is degenerate.
        ValueError: If ``core_min_weight``, ``n_init``, ``max_iter``,
            or ``tol`` violates its allowed range.

    Notes:
        The Akaike and Bayesian information criteria are computed using
        four free mixture parameters. They are intended for relative
        model comparison rather than as absolute goodness-of-fit measures.

        The Kolmogorov-Smirnov statistic is descriptive because the
        fitted parameters are estimated from the same catalogue.

        The reported ``outlier_frac`` is a mixture-model parameter
        and does not necessarily correspond to a uniquely identifiable
        population of catastrophic photometric-redshift outliers.

        The fitted ``outlier_mean_shift`` is converted into Binny's
        ``outlier_mean_scale`` and ``outlier_mean_offset`` by
        ``to_binny_photoz_params``.
    """
    z_true = np.asarray(z_true, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)

    if z_true.ndim != 1 or z_obs.shape != z_true.shape:
        raise ValueError("z_true and z_obs must be matching 1D arrays.")

    if z_true.size < 2 or not np.all(np.isfinite(z_true)) or not np.all(np.isfinite(z_obs)):
        raise ValueError("Redshifts must contain at least two finite values.")

    if np.any(z_true <= -1.0):
        raise ValueError("z_true must be greater than -1.")

    if not 0.0 < core_min_weight < 1.0:
        raise ValueError("core_min_weight must lie between 0 and 1.")

    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive.")

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite.")

    eps = (z_obs - z_true) / (1.0 + z_true)

    if not np.all(np.isfinite(eps)) or np.std(eps) == 0.0:
        raise ValueError("Residuals must be finite and non-degenerate.")

    sigma, weights, mu, ll, converged, n_iter = _fit_gaussian_mixture(
        eps,
        core_min_weight=core_min_weight,
        n_init=n_init,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
    )

    n = eps.size
    k = 4

    params = {
        "scatter_scale": float(sigma[0]),
        "outlier_scatter_scale": float(sigma[1]),
        "outlier_mean_shift": float(mu[1]),
        "outlier_frac": float(weights[1]),
    }

    if not return_metadata:
        return params

    n = eps.size
    k = 4

    metadata = {
        "loglike": ll,
        "n": int(n),
        "n_outside": 0,
        "converged": converged,
        "n_iter": n_iter,
        "aic": float(2 * k - 2 * ll),
        "bic": float(k * np.log(n) - 2 * ll),
        "ks_statistic": _ks_statistic(eps, sigma, mu, weights),
    }

    return params, metadata


def to_binny_photoz_params(fit: dict) -> dict[str, float]:
    """Convert fitted parameters to Binny's photometric-redshift assignment model.

    Converts the fitted Gaussian-mixture parameters into the complete
    parameter dictionary expected by Binny's photometric-redshift
    assignment model.

    The fitted core has zero mean and scatter ``scatter_scale``.
    The displaced tail has scatter ``outlier_scatter_scale``,
    displacement ``outlier_mean_shift``, and mixture fraction
    ``outlier_frac``.

    The tail displacement is represented in Binny using

        outlier_mean_scale = 1 + outlier_mean_shift
        outlier_mean_offset = -outlier_mean_shift.

    Args:
        fit: Dictionary containing ``scatter_scale``,
            ``outlier_scatter_scale``, ``outlier_mean_shift``, and
            ``outlier_frac``. Additional diagnostic entries are ignored.

    Returns:
        Dictionary containing the photometric-redshift assignment parameters:

            - mean_scale: Core mean scale, fixed at 1.
            - mean_offset: Core mean offset, fixed at 0.
            - scatter_scale: Fitted core scatter.
            - outlier_frac: Fitted tail mixture fraction.
            - outlier_mean_scale: Tail mean scale, equal to
              ``1 + outlier_mean_shift``.
            - outlier_mean_offset: Tail mean offset, equal to
              ``-outlier_mean_shift``.
            - outlier_scatter_scale: Fitted tail scatter.

    Raises:
        ValueError: If ``outlier_mean_shift`` is less than or equal
            to -1, making the corresponding mean scale non-positive.
        KeyError: If a required fitted parameter is missing.

    Notes:
        The conversion preserves the fitted normalized residual model

            epsilon = (z_obs - z_true) / (1 + z_true)

        when used with Binny's photometric-redshift assignment
        probability function.
    """
    shift = fit["outlier_mean_shift"]

    if shift <= -1.0:
        raise ValueError("Tail displacement is incompatible with Binny's mean scale.")

    return {
        "mean_scale": 1.0,
        "mean_offset": 0.0,
        "scatter_scale": fit["scatter_scale"],
        "outlier_frac": fit["outlier_frac"],
        "outlier_mean_scale": 1.0 + shift,
        "outlier_mean_offset": -shift,
        "outlier_scatter_scale": fit["outlier_scatter_scale"],
    }
