"""Unit tests for ``binny.calibration.photoz_fitting``."""

from __future__ import annotations

import numpy as np
import pytest

from binny.binning.photoz import photoz_assignment_probability
from binny.calibration.photoz_fitting import (
    _log_component_probability,
    _log_probability,
    fit_photoz_errors,
    photoz_residual_cdf,
    photoz_residual_pdf,
    sample_photoz_errors,
    to_binny_photoz_params,
)


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
    """Tests that component and mixture log probabilities agree."""
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
    """Tests that fitting recovers the injected core scatter."""
    assert fitted_catalogue["scatter_scale"] == pytest.approx(0.03, abs=0.005)


def test_photoz_fit_recovers_tail_parameters(fitted_catalogue):
    """Tests that fitting recovers the injected tail parameters."""
    assert fitted_catalogue["outlier_scatter_scale"] == pytest.approx(0.08, abs=0.015)
    assert fitted_catalogue["outlier_mean_shift"] == pytest.approx(0.12, abs=0.015)
    assert fitted_catalogue["outlier_frac"] == pytest.approx(0.20, abs=0.02)


def test_photoz_fit_returns_expected_keys(fitted_catalogue):
    """Tests that fitting returns the expected parameter keys."""
    assert set(fitted_catalogue) == {
        "scatter_scale",
        "outlier_scatter_scale",
        "outlier_mean_shift",
        "outlier_frac",
    }


def test_photoz_fit_returns_metadata(fitted_catalogue_with_metadata):
    """Tests that fitting returns the expected metadata."""
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
    """Tests that requesting metadata preserves fitted parameters."""
    params, _ = fitted_catalogue_with_metadata

    assert params == fitted_catalogue


def test_photoz_fit_returns_finite_parameters(
    fitted_catalogue,
    fitted_catalogue_with_metadata,
):
    """Tests that fitted parameters and diagnostics are finite."""
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
    """Tests that fitting uses the complete catalogue."""
    z_true, _ = synthetic_catalogue
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["n"] == len(z_true)
    assert metadata["n_outside"] == 0


def test_photoz_fit_uses_all_residuals():
    """Tests that fitting retains large residuals."""
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
    """Tests that expectation-maximization converges for synthetic data."""
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["converged"]
    assert 0 < metadata["n_iter"] <= 1000


def test_photoz_fit_information_criteria(fitted_catalogue_with_metadata):
    """Tests that information criteria match their definitions."""
    _, metadata = fitted_catalogue_with_metadata
    k = 4

    assert metadata["aic"] == pytest.approx(2 * k - 2 * metadata["loglike"])
    assert metadata["bic"] == pytest.approx(k * np.log(metadata["n"]) - 2 * metadata["loglike"])


def test_photoz_fit_goodness_of_fit(fitted_catalogue_with_metadata):
    """Tests that the fitted model describes synthetic residuals."""
    _, metadata = fitted_catalogue_with_metadata

    assert metadata["ks_statistic"] < 0.02


def test_photoz_fit_is_reproducible(synthetic_catalogue):
    """Tests that identical random seeds reproduce fitted parameters."""
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
    """Tests that identical random seeds reproduce fitting metadata."""
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
    """Tests that fitting respects the minimum core weight."""
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
    """Tests that fitting respects the maximum iteration count."""
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
    """Tests that invalid catalogue arrays raise ValueError."""
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
    """Tests that invalid fitting settings raise ValueError."""
    rng = np.random.default_rng(42)
    z_true = np.linspace(0.1, 1.0, 100)
    z_obs = z_true + rng.normal(0.0, 0.03, 100) * (1.0 + z_true)

    with pytest.raises(ValueError, match=message):
        fit_photoz_errors(z_true, z_obs, **kwargs)


def test_photoz_fit_rejects_degenerate_residuals():
    """Tests that degenerate residuals are rejected."""
    z_true = np.ones(100)
    z_obs = z_true + 0.02 * (1.0 + z_true)

    with pytest.raises(ValueError):
        fit_photoz_errors(z_true, z_obs)


def test_photoz_fit_rejects_obsolete_histogram_settings():
    """Tests that obsolete histogram settings are rejected."""
    rng = np.random.default_rng(42)
    z_true = np.ones(100)
    z_obs = z_true + rng.normal(0.0, 0.03, 100) * (1.0 + z_true)

    with pytest.raises(TypeError):
        fit_photoz_errors(z_true, z_obs, hist_range=0.3)

    with pytest.raises(TypeError):
        fit_photoz_errors(z_true, z_obs, hist_bin=0.01)


def test_photoz_mapping_returns_expected_parameters():
    """Tests that fitted parameters map to the Binny photo-z model."""
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
    """Tests that invalid tail displacements are rejected."""
    fit = {
        "scatter_scale": 0.03,
        "outlier_scatter_scale": 0.08,
        "outlier_mean_shift": -1.0,
        "outlier_frac": 0.20,
    }

    with pytest.raises(ValueError, match="Tail displacement"):
        to_binny_photoz_params(fit)


def test_photoz_mapping_works_with_forward_model():
    """Tests that mapped parameters work with the forward model."""
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
    """Tests that fitted parameters reproduce tomographic distributions."""
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


@pytest.fixture
def photoz_mixture_params() -> dict[str, float]:
    """Return reference Gaussian core and tail parameters."""
    return {
        "scatter_scale": 0.03,
        "outlier_scatter_scale": 0.08,
        "outlier_mean_shift": 0.12,
        "outlier_frac": 0.20,
    }


def test_photoz_residual_pdf_returns_expected_shape(photoz_mixture_params):
    """Tests that the residual PDF preserves the input shape."""
    x = np.linspace(-0.5, 0.5, 100)

    pdf = photoz_residual_pdf(x, **photoz_mixture_params)

    assert pdf.shape == x.shape
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0.0)


def test_photoz_residual_pdf_is_normalized(photoz_mixture_params):
    """Tests that the residual PDF integrates to unity."""
    x = np.linspace(-2.0, 2.0, 10_000)

    pdf = photoz_residual_pdf(x, **photoz_mixture_params)

    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-8)


def test_photoz_residual_pdf_returns_components(photoz_mixture_params):
    """Tests that the weighted component densities reproduce the total PDF."""
    x = np.linspace(-0.5, 0.5, 100)

    pdf, components = photoz_residual_pdf(
        x,
        **photoz_mixture_params,
        return_components=True,
    )

    assert set(components) == {"core", "outlier"}
    assert components["core"].shape == x.shape
    assert components["outlier"].shape == x.shape

    np.testing.assert_allclose(
        pdf,
        components["core"] + components["outlier"],
    )


def test_photoz_residual_pdf_component_normalization(photoz_mixture_params):
    """Tests that the component integrals match their mixture weights."""
    x = np.linspace(-2.0, 2.0, 10_000)

    _, components = photoz_residual_pdf(
        x,
        **photoz_mixture_params,
        return_components=True,
    )

    frac = photoz_mixture_params["outlier_frac"]

    assert np.trapezoid(components["core"], x) == pytest.approx(
        1.0 - frac,
        abs=1e-8,
    )
    assert np.trapezoid(components["outlier"], x) == pytest.approx(
        frac,
        abs=1e-8,
    )


def test_photoz_residual_pdf_matches_analytic_model(photoz_mixture_params):
    """Tests that the residual PDF matches the analytic Gaussian mixture."""
    x = np.linspace(-0.5, 0.5, 100)

    sigma_core = photoz_mixture_params["scatter_scale"]
    sigma_outlier = photoz_mixture_params["outlier_scatter_scale"]
    mu = photoz_mixture_params["outlier_mean_shift"]
    frac = photoz_mixture_params["outlier_frac"]

    core = (1.0 - frac) * np.exp(-0.5 * (x / sigma_core) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_core)
    outlier = (
        frac
        * np.exp(-0.5 * ((x - mu) / sigma_outlier) ** 2)
        / (np.sqrt(2.0 * np.pi) * sigma_outlier)
    )

    pdf = photoz_residual_pdf(x, **photoz_mixture_params)

    np.testing.assert_allclose(pdf, core + outlier)


def test_photoz_residual_pdf_preserves_local_density(photoz_mixture_params):
    """Tests that the residual PDF is not renormalized over the input grid."""
    x = np.linspace(-0.5, 0.5, 1000)

    pdf = photoz_residual_pdf(x, **photoz_mixture_params)
    subset = photoz_residual_pdf(x[400:600], **photoz_mixture_params)

    np.testing.assert_allclose(subset, pdf[400:600])


@pytest.mark.parametrize("outlier_frac", [0.0, 1.0])
def test_photoz_residual_pdf_accepts_boundary_weights(
    photoz_mixture_params,
    outlier_frac,
):
    """Tests that zero and unit outlier fractions are supported."""
    params = {**photoz_mixture_params, "outlier_frac": outlier_frac}
    x = np.linspace(-2.0, 2.0, 10_000)

    pdf, components = photoz_residual_pdf(
        x,
        **params,
        return_components=True,
    )

    assert np.all(np.isfinite(pdf))
    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-8)

    inactive = "outlier" if outlier_frac == 0.0 else "core"

    np.testing.assert_array_equal(components[inactive], 0.0)


def test_photoz_residual_pdf_accepts_fitted_parameters(fitted_catalogue):
    """Tests that fitted parameters can be passed directly to the PDF."""
    x = np.linspace(-2.0, 2.0, 10_000)

    pdf = photoz_residual_pdf(x, **fitted_catalogue)

    assert np.all(np.isfinite(pdf))
    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-8)


def test_photoz_residual_cdf_returns_expected_shape(photoz_mixture_params):
    """Tests that the residual CDF preserves the input shape."""
    x = np.linspace(-0.5, 0.5, 100)

    cdf = photoz_residual_cdf(x, **photoz_mixture_params)

    assert cdf.shape == x.shape
    assert np.all(np.isfinite(cdf))


def test_photoz_residual_cdf_is_bounded(photoz_mixture_params):
    """Tests that cumulative probabilities lie between zero and one."""
    x = np.linspace(-2.0, 2.0, 1000)

    cdf = photoz_residual_cdf(x, **photoz_mixture_params)

    assert np.all((cdf >= 0.0) & (cdf <= 1.0))


def test_photoz_residual_cdf_is_monotonic(photoz_mixture_params):
    """Tests that the residual CDF is monotonically non-decreasing."""
    x = np.linspace(-2.0, 2.0, 1000)

    cdf = photoz_residual_cdf(x, **photoz_mixture_params)

    assert np.all(np.diff(cdf) >= 0.0)


def test_photoz_residual_cdf_has_correct_limits(photoz_mixture_params):
    """Tests that the residual CDF has the correct limits at infinity."""
    x = np.array([-np.inf, np.inf])

    cdf = photoz_residual_cdf(x, **photoz_mixture_params)

    np.testing.assert_allclose(cdf, [0.0, 1.0])


@pytest.mark.parametrize("outlier_frac", [0.0, 1.0])
def test_photoz_residual_cdf_accepts_boundary_weights(
    photoz_mixture_params,
    outlier_frac,
):
    """Tests that the CDF supports zero and unit outlier fractions."""
    params = {**photoz_mixture_params, "outlier_frac": outlier_frac}

    if outlier_frac == 0.0:
        center = params["scatter_scale"] * 0.0
    else:
        center = params["outlier_mean_shift"]

    x = np.array([-np.inf, center, np.inf])

    cdf = photoz_residual_cdf(x, **params)

    np.testing.assert_allclose(cdf, [0.0, 0.5, 1.0])


def test_photoz_residual_cdf_derivative_matches_pdf(photoz_mixture_params):
    """Tests that differentiating the residual CDF reproduces the PDF."""
    x = np.linspace(-0.3, 0.5, 10_000)

    cdf = photoz_residual_cdf(x, **photoz_mixture_params)
    pdf = photoz_residual_pdf(x, **photoz_mixture_params)

    derivative = np.gradient(cdf, x)

    np.testing.assert_allclose(
        derivative[1:-1],
        pdf[1:-1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_photoz_residual_cdf_accepts_fitted_parameters(fitted_catalogue):
    """Tests that fitted parameters can be passed directly to the CDF."""
    x = np.array([-np.inf, 0.0, np.inf])

    cdf = photoz_residual_cdf(x, **fitted_catalogue)

    assert cdf[0] == pytest.approx(0.0)
    assert 0.0 < cdf[1] < 1.0
    assert cdf[-1] == pytest.approx(1.0)


def test_photoz_residual_cdf_matches_forward_model(photoz_mixture_params):
    """Tests that residual CDF differences reproduce assignment probabilities."""
    z = np.linspace(0.0, 2.0, 100)
    z_min = 0.5
    z_max = 1.0

    lower = (z_min - z) / (1.0 + z)
    upper = (z_max - z) / (1.0 + z)

    probability = photoz_residual_cdf(upper, **photoz_mixture_params) - photoz_residual_cdf(
        lower, **photoz_mixture_params
    )

    expected = photoz_assignment_probability(
        z,
        bin_min=z_min,
        bin_max=z_max,
        **to_binny_photoz_params(photoz_mixture_params),
    )

    np.testing.assert_allclose(
        probability,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_sample_photoz_errors_returns_expected_shape(photoz_mixture_params):
    """Tests that sampling preserves the input redshift shape."""
    z_true = np.linspace(0.1, 2.0, 1000)

    z_obs = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )

    assert z_obs.shape == z_true.shape
    assert np.all(np.isfinite(z_obs))


def test_sample_photoz_errors_seed_is_reproducible(photoz_mixture_params):
    """Tests that identical random seeds reproduce sampled redshifts."""
    z_true = np.linspace(0.1, 2.0, 1000)

    first = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )
    second = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_sample_photoz_errors_different_seeds(photoz_mixture_params):
    """Tests that different random seeds produce different redshifts."""
    z_true = np.linspace(0.1, 2.0, 1000)

    first = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )
    second = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=43,
    )

    assert not np.array_equal(first, second)


def test_sample_photoz_errors_accepts_rng(photoz_mixture_params):
    """Tests that sampling supports a NumPy random number generator."""
    z_true = np.linspace(0.1, 2.0, 1000)

    first = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        rng=np.random.default_rng(42),
    )
    second = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first, second)


def test_sample_photoz_errors_advances_rng(photoz_mixture_params):
    """Tests that sampling advances the supplied random generator."""
    z_true = np.linspace(0.1, 2.0, 1000)
    rng = np.random.default_rng(42)

    first = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        rng=rng,
    )
    second = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        rng=rng,
    )

    assert not np.array_equal(first, second)


def test_sample_photoz_errors_without_seed(photoz_mixture_params):
    """Tests that sampling works without an explicit random seed."""
    z_true = np.linspace(0.1, 2.0, 100)

    z_obs = sample_photoz_errors(z_true, **photoz_mixture_params)

    assert z_obs.shape == z_true.shape
    assert np.all(np.isfinite(z_obs))


@pytest.mark.parametrize("outlier_frac", [0.0, 1.0])
def test_sample_photoz_errors_recovers_single_component(
    photoz_mixture_params,
    outlier_frac,
):
    """Tests that sampling reproduces the active Gaussian component."""
    params = {**photoz_mixture_params, "outlier_frac": outlier_frac}
    z_true = np.linspace(0.1, 2.0, 100_000)

    z_obs = sample_photoz_errors(z_true, **params, seed=42)
    eps = (z_obs - z_true) / (1.0 + z_true)

    if outlier_frac == 0.0:
        expected_mean = 0.0
        expected_sigma = params["scatter_scale"]
    else:
        expected_mean = params["outlier_mean_shift"]
        expected_sigma = params["outlier_scatter_scale"]

    assert np.mean(eps) == pytest.approx(expected_mean, abs=0.001)
    assert np.std(eps) == pytest.approx(expected_sigma, rel=0.02)


def test_sample_photoz_errors_recovers_mixture_moments(photoz_mixture_params):
    """Tests that sampled residual moments match the analytic mixture moments."""
    z_true = np.linspace(0.1, 2.0, 100_000)

    z_obs = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )
    eps = (z_obs - z_true) / (1.0 + z_true)

    frac = photoz_mixture_params["outlier_frac"]
    sigma_core = photoz_mixture_params["scatter_scale"]
    sigma_outlier = photoz_mixture_params["outlier_scatter_scale"]
    mu = photoz_mixture_params["outlier_mean_shift"]

    expected_mean = frac * mu
    expected_variance = (
        (1.0 - frac) * sigma_core**2 + frac * sigma_outlier**2 + frac * (1.0 - frac) * mu**2
    )

    assert np.mean(eps) == pytest.approx(expected_mean, abs=0.001)
    assert np.var(eps) == pytest.approx(expected_variance, rel=0.03)


def test_sample_photoz_errors_matches_residual_cdf(photoz_mixture_params):
    """Tests that sampled residuals reproduce the analytic residual CDF."""
    z_true = np.linspace(0.1, 2.0, 100_000)

    z_obs = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )
    eps = np.sort((z_obs - z_true) / (1.0 + z_true))

    model_cdf = photoz_residual_cdf(eps, **photoz_mixture_params)

    n = eps.size
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(n) / n

    statistic = max(
        np.max(empirical_upper - model_cdf),
        np.max(model_cdf - empirical_lower),
    )

    assert statistic < 0.01


def test_sample_photoz_errors_preserves_input(photoz_mixture_params):
    """Tests that sampling does not modify the input redshift array."""
    z_true = np.linspace(0.1, 2.0, 1000)
    original = z_true.copy()

    sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )

    np.testing.assert_array_equal(z_true, original)


def test_sample_photoz_errors_does_not_clip_redshifts(photoz_mixture_params):
    """Tests that negative sampled redshifts are not clipped."""
    params = {**photoz_mixture_params, "outlier_frac": 0.0}
    z_true = np.zeros(1000)

    z_obs = sample_photoz_errors(z_true, **params, seed=42)

    assert np.any(z_obs < 0.0)
    assert np.any(z_obs > 0.0)


def test_sample_photoz_errors_rejects_seed_and_rng(photoz_mixture_params):
    """Tests that supplying both seed and rng raises ValueError."""
    z_true = np.linspace(0.1, 2.0, 100)

    with pytest.raises(ValueError, match="seed"):
        sample_photoz_errors(
            z_true,
            **photoz_mixture_params,
            seed=42,
            rng=np.random.default_rng(42),
        )


def test_sample_photoz_errors_rejects_invalid_rng(photoz_mixture_params):
    """Tests that an invalid random number generator raises ValueError."""
    z_true = np.linspace(0.1, 2.0, 100)

    with pytest.raises(ValueError, match="rng"):
        sample_photoz_errors(
            z_true,
            **photoz_mixture_params,
            rng=42,
        )


@pytest.mark.parametrize(
    "z_true",
    [
        0.5,
        [[0.1, 0.2]],
        [0.1, np.nan],
        [0.1, np.inf],
        [0.1, -np.inf],
        [0.1, -1.0],
        [0.1, -1.5],
    ],
)
def test_sample_photoz_errors_rejects_invalid_redshifts(
    photoz_mixture_params,
    z_true,
):
    """Tests that invalid true redshifts raise ValueError."""
    with pytest.raises(ValueError):
        sample_photoz_errors(
            z_true,
            **photoz_mixture_params,
            seed=42,
        )


@pytest.mark.parametrize(
    "function",
    [
        photoz_residual_pdf,
        photoz_residual_cdf,
    ],
)
@pytest.mark.parametrize(
    "x",
    [
        0.0,
        [[0.0, 0.1]],
        [0.0, np.nan],
    ],
)
def test_photoz_residual_functions_reject_invalid_residuals(
    photoz_mixture_params,
    function,
    x,
):
    """Tests that invalid residual arrays raise ValueError."""
    with pytest.raises(ValueError):
        function(x, **photoz_mixture_params)


@pytest.mark.parametrize(
    "function",
    [
        photoz_residual_pdf,
        photoz_residual_cdf,
        sample_photoz_errors,
    ],
)
@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("scatter_scale", 0.0, "Scatter scales"),
        ("scatter_scale", -0.03, "Scatter scales"),
        ("scatter_scale", np.nan, "Scatter scales"),
        ("scatter_scale", np.inf, "Scatter scales"),
        ("outlier_scatter_scale", 0.0, "Scatter scales"),
        ("outlier_scatter_scale", -0.08, "Scatter scales"),
        ("outlier_scatter_scale", np.nan, "Scatter scales"),
        ("outlier_scatter_scale", np.inf, "Scatter scales"),
        ("outlier_mean_shift", np.nan, "Component means"),
        ("outlier_mean_shift", np.inf, "Component means"),
        ("outlier_mean_shift", -np.inf, "Component means"),
        ("outlier_frac", -0.1, "outlier_frac"),
        ("outlier_frac", 1.1, "outlier_frac"),
        ("outlier_frac", np.nan, "outlier_frac"),
        ("outlier_frac", np.inf, "outlier_frac"),
    ],
)
def test_photoz_mixture_functions_reject_invalid_parameters(
    photoz_mixture_params,
    function,
    parameter,
    value,
    message,
):
    """Tests that invalid Gaussian mixture parameters raise ValueError."""
    params = {**photoz_mixture_params, parameter: value}
    x = np.linspace(0.1, 1.0, 100)

    with pytest.raises(ValueError, match=message):
        function(x, **params)


def test_sample_photoz_errors_recovers_fitted_parameters(photoz_mixture_params):
    """Tests that fitting sampled redshifts recovers the injected parameters."""
    z_true = np.linspace(0.1, 2.0, 30_000)

    z_obs = sample_photoz_errors(
        z_true,
        **photoz_mixture_params,
        seed=42,
    )

    fit = fit_photoz_errors(
        z_true,
        z_obs,
        n_init=3,
        seed=42,
    )

    assert fit["scatter_scale"] == pytest.approx(
        photoz_mixture_params["scatter_scale"],
        abs=0.005,
    )
    assert fit["outlier_scatter_scale"] == pytest.approx(
        photoz_mixture_params["outlier_scatter_scale"],
        abs=0.015,
    )
    assert fit["outlier_mean_shift"] == pytest.approx(
        photoz_mixture_params["outlier_mean_shift"],
        abs=0.015,
    )
    assert fit["outlier_frac"] == pytest.approx(
        photoz_mixture_params["outlier_frac"],
        abs=0.03,
    )
