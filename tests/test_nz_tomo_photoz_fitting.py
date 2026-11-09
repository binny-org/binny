"""Unit tests for ``binny.nz_tomo.photoz_fitting``."""

from __future__ import annotations

import numpy as np
import pytest

from binny.binning.photoz_fitting import (
    _log_component_probability,
    _log_probability,
    fit_photoz_errors,
    to_binny_photoz_params,
)
from binny.nz_tomo.photoz import photoz_assignment_probability


@pytest.fixture(scope="module")
def synthetic_catalogue() -> tuple[np.ndarray, np.ndarray]:
    """Generate a catalogue with known Gaussian core and tail parameters."""
    rng = np.random.default_rng(42)
    n = 100_000

    z_true = rng.uniform(0.1, 2.0, n)
    outliers = rng.random(n) < 0.20

    eps = rng.normal(0.0, 0.03, n)
    eps[outliers] = rng.normal(0.12, 0.08, outliers.sum())

    z_obs = z_true + eps * (1.0 + z_true)
    return z_true, z_obs


@pytest.fixture(scope="module")
def fitted_catalogue(synthetic_catalogue) -> dict[str, float]:
    """Fit the shared synthetic catalogue once without metadata."""
    z_true, z_obs = synthetic_catalogue
    return fit_photoz_errors(z_true, z_obs, seed=42)


@pytest.fixture(scope="module")
def fitted_catalogue_with_metadata(
    synthetic_catalogue,
) -> tuple[dict[str, float], dict[str, float | int | bool]]:
    """Fit the shared synthetic catalogue once with metadata."""
    z_true, z_obs = synthetic_catalogue
    return fit_photoz_errors(
        z_true,
        z_obs,
        seed=42,
        return_metadata=True,
    )


def test_log_component_probability_matches_mixture():
    """Test consistency between component and mixture log probabilities."""
    from scipy.special import logsumexp

    x = np.array([-0.1, 0.0, 0.1])
    sigma = np.array([0.03, 0.08])
    mu = np.array([0.0, 0.12])
    weights = np.array([0.8, 0.2])

    logp = _log_component_probability(x, sigma, mu, weights)
    mixture = _log_probability(x, sigma, mu, weights)

    assert logp.shape == (3, 2)
    np.testing.assert_allclose(mixture, logsumexp(logp, axis=1))


def test_photoz_fit_recovers_core_scatter(fitted_catalogue):
    """Test recovery of the injected core scatter."""
    assert fitted_catalogue["scatter_scale"] == pytest.approx(0.03, abs=0.005)


def test_photoz_fit_recovers_tail_parameters(fitted_catalogue):
    """Test recovery of the injected tail parameters."""
    assert fitted_catalogue["outlier_scatter_scale"] == pytest.approx(0.08, abs=0.015)
    assert fitted_catalogue["outlier_mean_shift"] == pytest.approx(0.12, abs=0.015)
    assert fitted_catalogue["outlier_frac"] == pytest.approx(0.20, abs=0.02)


def test_photoz_fit_returns_expected_keys(fitted_catalogue):
    """Test that the default result contains only fitted parameters."""
    assert set(fitted_catalogue) == {
        "scatter_scale",
        "outlier_scatter_scale",
        "outlier_mean_shift",
        "outlier_frac",
    }


def test_photoz_fit_returns_metadata(fitted_catalogue_with_metadata):
    """Test that optional metadata contains all fitting diagnostics."""
    params, metadata = fitted_catalogue_with_metadata

    assert set(params) == {
        "scatter_scale",
        "outlier_scatter_scale",
        "outlier_mean_shift",
        "outlier_frac",
    }

    assert set(metadata) == {
        "loglike",
        "n",
        "n_outside",
        "converged",
        "n_iter",
        "aic",
        "bic",
        "ks_statistic",
    }


def test_photoz_fit_metadata_preserves_parameters(
    fitted_catalogue,
    fitted_catalogue_with_metadata,
):
    """Test that requesting metadata does not change fitted parameters."""
    params, _ = fitted_catalogue_with_metadata

    assert params == fitted_catalogue


def test_photoz_fit_returns_finite_parameters(
    fitted_catalogue,
    fitted_catalogue_with_metadata,
):
    """Test that fitted parameters and diagnostics are finite."""
    _, metadata = fitted_catalogue_with_metadata

    assert all(np.isfinite(value) for value in fitted_catalogue.values())

    assert fitted_catalogue["scatter_scale"] > 0.0
    assert fitted_catalogue["outlier_scatter_scale"] > 0.0
    assert 0.0 <= fitted_catalogue["outlier_frac"] <= 0.5

    assert np.isfinite(metadata["loglike"])
    assert np.isfinite(metadata["aic"])
    assert np.isfinite(metadata["bic"])
    assert metadata["n_iter"] > 0
    assert 0.0 <= metadata["ks_statistic"] <= 1.0


def test_photoz_fit_counts_catalogue_objects(
    synthetic_catalogue,
    fitted_catalogue_with_metadata,
):
    """Test that the full catalogue is used in the fit."""
    z_true, _ = synthetic_catalogue
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["n"] == len(z_true)
    assert metadata["n_outside"] == 0


def test_photoz_fit_uses_all_residuals():
    """Test that large residuals are retained in the unbinned fit."""
    rng = np.random.default_rng(123)
    n = 1000

    z_true = np.ones(n)
    eps = rng.normal(0.0, 0.03, n)
    eps[:20] = rng.normal(0.6, 0.05, 20)

    z_obs = z_true + eps * (1.0 + z_true)

    _, metadata = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=123,
        return_metadata=True,
    )

    assert metadata["n"] == n
    assert metadata["n_outside"] == 0
    assert np.isfinite(metadata["loglike"])


def test_photoz_fit_converges(fitted_catalogue_with_metadata):
    """Test that expectation-maximization converges for synthetic data."""
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["converged"]
    assert 0 < metadata["n_iter"] <= 1000


def test_photoz_fit_information_criteria(fitted_catalogue_with_metadata):
    """Test the Akaike and Bayesian information criteria."""
    _, metadata = fitted_catalogue_with_metadata
    k = 4

    assert metadata["aic"] == pytest.approx(2 * k - 2 * metadata["loglike"])
    assert metadata["bic"] == pytest.approx(k * np.log(metadata["n"]) - 2 * metadata["loglike"])


def test_photoz_fit_goodness_of_fit(fitted_catalogue_with_metadata):
    """Test that the fitted model describes the synthetic residuals."""
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["ks_statistic"] < 0.02


def test_photoz_fit_is_reproducible(synthetic_catalogue):
    """Test that identical random seeds produce identical results."""
    z_true, z_obs = synthetic_catalogue

    fit1 = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=42,
    )
    fit2 = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=42,
    )

    assert fit1 == fit2


def test_photoz_fit_metadata_is_reproducible(synthetic_catalogue):
    """Test reproducibility when fitting diagnostics are requested."""
    z_true, z_obs = synthetic_catalogue

    result1 = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=42,
        return_metadata=True,
    )
    result2 = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=42,
        return_metadata=True,
    )

    assert result1 == result2


def test_photoz_fit_respects_core_min_weight(synthetic_catalogue):
    """Test the lower bound imposed on the core mixture weight."""
    z_true, z_obs = synthetic_catalogue

    fit = fit_photoz_errors(
        z_true,
        z_obs,
        core_min_weight=0.9,
        n_init=3,
        seed=42,
    )

    assert fit["outlier_frac"] <= 0.1 + 1e-10


def test_photoz_fit_respects_max_iter(synthetic_catalogue):
    """Test that the optimizer respects the iteration limit."""
    z_true, z_obs = synthetic_catalogue

    _, metadata = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=1,
        max_iter=2,
        seed=42,
        return_metadata=True,
    )

    assert metadata["n_iter"] <= 2


@pytest.mark.parametrize(
    ("z_true", "z_obs"),
    [
        ([], []),
        ([0.1, 0.2], [0.1]),
        ([0.1, np.nan], [0.2, 0.3]),
        ([0.1, np.inf], [0.2, 0.3]),
        ([0.1, 0.2], [0.2, np.nan]),
        ([-1.0, 0.2], [0.1, 0.3]),
        ([[0.1, 0.2]], [[0.2, 0.3]]),
    ],
)
def test_photoz_fit_rejects_invalid_catalogues(z_true, z_obs):
    """Test that invalid catalogue arrays raise ValueError."""
    with pytest.raises(ValueError):
        fit_photoz_errors(z_true, z_obs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"core_min_weight": 0.0}, "core_min_weight"),
        ({"core_min_weight": 1.0}, "core_min_weight"),
        ({"n_init": 0}, "n_init"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"tol": -1e-8}, "tol"),
    ],
)
def test_photoz_fit_rejects_invalid_settings(kwargs, message):
    """Test validation of unbinned expectation-maximization settings."""
    rng = np.random.default_rng(42)
    z_true = np.linspace(0.1, 1.0, 100)
    z_obs = z_true + rng.normal(0.0, 0.03, 100) * (1.0 + z_true)

    with pytest.raises(ValueError, match=message):
        fit_photoz_errors(z_true, z_obs, **kwargs)


def test_photoz_fit_rejects_degenerate_residuals():
    """Test that constant residuals cannot define a mixture fit."""
    z_true = np.ones(100)
    z_obs = z_true + 0.02 * (1.0 + z_true)

    with pytest.raises(ValueError):
        fit_photoz_errors(z_true, z_obs)


def test_photoz_fit_rejects_obsolete_histogram_settings():
    """Test that histogram controls are no longer part of the public interface."""
    rng = np.random.default_rng(42)
    z_true = np.ones(100)
    z_obs = z_true + rng.normal(0.0, 0.03, 100) * (1.0 + z_true)

    with pytest.raises(TypeError):
        fit_photoz_errors(z_true, z_obs, hist_range=0.3)

    with pytest.raises(TypeError):
        fit_photoz_errors(z_true, z_obs, hist_bin=0.01)


def test_photoz_mapping_returns_expected_parameters():
    """Test conversion to Binny's photometric-redshift parameters."""
    fit = {
        "scatter_scale": 0.03,
        "outlier_scatter_scale": 0.08,
        "outlier_mean_shift": 0.12,
        "outlier_frac": 0.20,
    }

    params = to_binny_photoz_params(fit)

    assert params == {
        "mean_scale": 1.0,
        "mean_offset": 0.0,
        "scatter_scale": 0.03,
        "outlier_frac": 0.20,
        "outlier_mean_scale": 1.12,
        "outlier_mean_offset": -0.12,
        "outlier_scatter_scale": 0.08,
    }


def test_photoz_mapping_rejects_invalid_tail_mean():
    """Test rejection of incompatible tail displacements."""
    fit = {
        "scatter_scale": 0.03,
        "outlier_scatter_scale": 0.08,
        "outlier_mean_shift": -1.0,
        "outlier_frac": 0.20,
    }

    with pytest.raises(ValueError, match="Tail displacement"):
        to_binny_photoz_params(fit)


def test_photoz_mapping_works_with_forward_model():
    """Test that fitted parameters work with the photometric-redshift model."""
    fit = {
        "scatter_scale": 0.03,
        "outlier_scatter_scale": 0.08,
        "outlier_mean_shift": 0.12,
        "outlier_frac": 0.20,
    }

    params = to_binny_photoz_params(fit)
    z = np.linspace(0.0, 2.0, 100)

    probability = photoz_assignment_probability(
        z,
        bin_min=0.5,
        bin_max=1.0,
        **params,
    )

    assert probability.shape == z.shape
    assert np.all(np.isfinite(probability))
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_photoz_fit_recovers_tomographic_bins(fitted_catalogue):
    """Test that fitted photometric-redshift errors reproduce tomography."""
    fitted_params = to_binny_photoz_params(fitted_catalogue)

    true_params = to_binny_photoz_params(
        {
            "scatter_scale": 0.03,
            "outlier_scatter_scale": 0.08,
            "outlier_mean_shift": 0.12,
            "outlier_frac": 0.20,
        }
    )

    z = np.linspace(0.0, 3.0, 1000)
    nz = z**2 * np.exp(-((z / 0.8) ** 1.5))
    nz /= np.trapezoid(nz, z)

    edges = np.linspace(0.2, 2.2, 6)

    for z_min, z_max in zip(edges[:-1], edges[1:], strict=True):
        p_true = photoz_assignment_probability(
            z,
            bin_min=z_min,
            bin_max=z_max,
            **true_params,
        )
        p_fit = photoz_assignment_probability(
            z,
            bin_min=z_min,
            bin_max=z_max,
            **fitted_params,
        )

        nz_true = nz * p_true
        nz_fit = nz * p_fit

        n_true = np.trapezoid(nz_true, z)
        n_fit = np.trapezoid(nz_fit, z)

        assert n_fit == pytest.approx(n_true, rel=0.05)

        mean_true = np.trapezoid(z * nz_true, z) / n_true
        mean_fit = np.trapezoid(z * nz_fit, z) / n_fit

        assert mean_fit == pytest.approx(mean_true, abs=0.02)

        nz_true /= n_true
        nz_fit /= n_fit

        difference = np.trapezoid(np.abs(nz_fit - nz_true), z)

        assert difference < 0.10
