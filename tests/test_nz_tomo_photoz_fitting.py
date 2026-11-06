"""Unit tests for "binny.nz_tomo.photoz_fitting"."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.photoz import photoz_assignment_probability
from binny.nz_tomo.photoz_fitting import (
    fit_photoz_errors,
    to_binny_photoz_params,
)


@pytest.fixture
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


def test_photoz_fit_recovers_core_scatter(synthetic_catalogue):
    """Tests that the fitted core scatter matches the injected value."""
    z_true, z_obs = synthetic_catalogue
    fit = fit_photoz_errors(z_true, z_obs)

    assert fit["sigma_c"] == pytest.approx(0.03, abs=0.005)


def test_photoz_fit_recovers_tail_parameters(synthetic_catalogue):
    """Tests that the fitted tail parameters match injected values."""
    z_true, z_obs = synthetic_catalogue
    fit = fit_photoz_errors(z_true, z_obs)

    assert fit["sigma_t"] == pytest.approx(0.08, abs=0.015)
    assert fit["m_t"] == pytest.approx(0.12, abs=0.015)
    assert fit["f_t"] == pytest.approx(0.20, abs=0.05)


def test_photoz_fit_returns_expected_keys(synthetic_catalogue):
    """Tests that the fit returns all expected parameters and diagnostics."""
    z_true, z_obs = synthetic_catalogue
    fit = fit_photoz_errors(z_true, z_obs)

    assert set(fit) == {
        "sigma_c",
        "sigma_t",
        "m_t",
        "f_t",
        "loglike",
        "n",
        "n_outside",
    }


def test_photoz_fit_returns_finite_parameters(synthetic_catalogue):
    """Tests that all fitted parameters and the likelihood are finite."""
    z_true, z_obs = synthetic_catalogue
    fit = fit_photoz_errors(z_true, z_obs)

    assert all(np.isfinite(value) for value in fit.values())
    assert fit["sigma_c"] > 0.0
    assert fit["sigma_t"] > 0.0
    assert 0.0 <= fit["f_t"] <= 0.5


def test_photoz_fit_counts_catalogue_objects(synthetic_catalogue):
    """Tests that the fit reports the total catalogue size."""
    z_true, z_obs = synthetic_catalogue
    fit = fit_photoz_errors(z_true, z_obs)

    assert fit["n"] == len(z_true)


def test_photoz_fit_counts_outside_residuals():
    """Tests that residuals outside the fitting range are counted."""
    z_true = np.ones(6)
    eps = np.array([-0.5, -0.02, 0.01, 0.03, 0.04, 0.5])
    z_obs = z_true + eps * (1.0 + z_true)

    fit = fit_photoz_errors(z_true, z_obs, hist_bin=0.01)

    assert fit["n"] == 6
    assert fit["n_outside"] == 2


def test_photoz_fit_is_reproducible(synthetic_catalogue):
    """Tests that identical seeds produce identical fitted results."""
    z_true, z_obs = synthetic_catalogue

    fit1 = fit_photoz_errors(z_true, z_obs, seed=42)
    fit2 = fit_photoz_errors(z_true, z_obs, seed=42)

    assert fit1 == fit2


def test_photoz_fit_respects_core_min_weight(synthetic_catalogue):
    """Tests that the fitted core fraction respects the specified minimum."""
    z_true, z_obs = synthetic_catalogue

    fit = fit_photoz_errors(
        z_true,
        z_obs,
        core_min_weight=0.9,
    )

    assert fit["f_t"] <= 0.1 + 1e-10


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
    """Tests that invalid catalogue arrays raise ValueError."""
    with pytest.raises(ValueError):
        fit_photoz_errors(z_true, z_obs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hist_range": 0.0}, "histogram"),
        ({"hist_bin": 0.0}, "histogram"),
        ({"hist_bin": 1.0}, "histogram"),
        ({"core_min_weight": 0.0}, "core_min_weight"),
        ({"core_min_weight": 1.0}, "core_min_weight"),
        ({"n_init": 0}, "n_init"),
    ],
)
def test_photoz_fit_rejects_invalid_settings(kwargs, message):
    """Tests that invalid fitting settings raise ValueError."""
    z_true = np.linspace(0.1, 1.0, 100)
    z_obs = z_true + 0.01 * (1.0 + z_true)

    with pytest.raises(ValueError, match=message):
        fit_photoz_errors(z_true, z_obs, **kwargs)


def test_photoz_fit_rejects_insufficient_residuals():
    """Tests that insufficient histogram data raise ValueError."""
    z_true = np.ones(10)
    z_obs = np.full(10, 5.0)

    with pytest.raises(ValueError, match="Insufficient residuals"):
        fit_photoz_errors(z_true, z_obs)


def test_photoz_mapping_returns_expected_parameters():
    """Tests that fitted parameters map to Binny's photo-z convention."""
    fit = {
        "sigma_c": 0.03,
        "sigma_t": 0.08,
        "m_t": 0.12,
        "f_t": 0.20,
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
    """Tests that incompatible tail displacements raise ValueError."""
    fit = {
        "sigma_c": 0.03,
        "sigma_t": 0.08,
        "m_t": -1.0,
        "f_t": 0.20,
    }

    with pytest.raises(ValueError, match="Tail displacement"):
        to_binny_photoz_params(fit)


def test_photoz_mapping_works_with_forward_model():
    """Tests that mapped parameters work with Binny's photo-z model."""
    fit = {
        "sigma_c": 0.03,
        "sigma_t": 0.08,
        "m_t": 0.12,
        "f_t": 0.20,
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
