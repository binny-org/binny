"""Unit tests for ``binny.nz.models`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz.models import (
    gamma_distribution,
    gaussian_distribution,
    gaussian_mixture_distribution,
    lognormal_distribution,
    schechter_like_distribution,
    shifted_smail_distribution,
    skew_normal_distribution,
    smail_like_distribution,
    student_t_distribution,
    tabulated_distribution,
    tophat_distribution,
)


def _assert_allclose(a, b, *, rtol=1e-13, atol=1e-15):
    """Asserts that two arrays or scalars are approximately equal."""
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_smail_like_distribution_scalar_matches_formula():
    """Tests that smail_like_distribution matches the formula for a scalar input."""
    z = 1.2
    z0, alpha, beta = 0.5, 2.0, 1.7
    got = smail_like_distribution(z, z0=z0, alpha=alpha, beta=beta)
    expected = (z / z0) ** alpha * np.exp(-((z / z0) ** beta))
    _assert_allclose(got, expected)


def test_smail_like_distribution_vectorized():
    """Tests that smail_like_distribution works for array input."""
    z = np.linspace(0.0, 3.0, 7)
    z0, alpha, beta = 0.7, 1.2, 2.3
    got = smail_like_distribution(z, z0=z0, alpha=alpha, beta=beta)
    expected = (z / z0) ** alpha * np.exp(-((z / z0) ** beta))
    _assert_allclose(got, expected)
    assert got.shape == z.shape


def test_gaussian_distribution_peak_is_one_at_mu():
    """Tests that gaussian_distribution peaks at 1.0 when z=mu."""
    mu, sigma = 0.8, 0.2
    got = gaussian_distribution(mu, mu=mu, sigma=sigma)
    _assert_allclose(got, 1.0)


def test_gaussian_distribution_is_symmetric_about_mu():
    """Tests that gaussian_distribution is symmetric about mu."""
    mu, sigma, d = 1.0, 0.3, 0.17
    left = gaussian_distribution(mu - d, mu=mu, sigma=sigma)
    right = gaussian_distribution(mu + d, mu=mu, sigma=sigma)
    _assert_allclose(left, right)


@pytest.mark.parametrize("sigma", [0.0, -1e-6, -2.0])
def test_gaussian_distribution_raises_for_nonpositive_sigma(sigma):
    """Tests that gaussian_distribution raises ValueError for non-positive sigma."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_distribution(0.1, mu=0.0, sigma=sigma)


def test_gaussian_mixture_equal_weights_matches_manual_sum():
    """Tests that gaussian_mixture_distribution with equal weights"""
    z = np.array([0.0, 0.5, 1.0])
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.2, 0.4])

    got = gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas, weights=None)

    g0 = np.exp(-0.5 * ((z - mus[0]) / sigmas[0]) ** 2)
    g1 = np.exp(-0.5 * ((z - mus[1]) / sigmas[1]) ** 2)
    expected = g0 + g1
    _assert_allclose(got, expected)


def test_gaussian_mixture_with_weights_matches_manual_sum():
    """Tests that gaussian_mixture_distribution with specified weights"""
    z = np.linspace(-1.0, 2.0, 10)
    mus = np.array([0.0, 1.0, 1.5])
    sigmas = np.array([0.3, 0.2, 0.7])
    w = np.array([2.0, 0.5, 3.0])

    got = gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas, weights=w)

    expected = np.zeros_like(z, dtype=float)
    for wi, mi, si in zip(w, mus, sigmas, strict=True):
        expected += wi * np.exp(-0.5 * ((z - mi) / si) ** 2)

    _assert_allclose(got, expected)


def test_gaussian_mixture_preserves_shape_for_array_input():
    """Tests that gaussian_mixture_distribution preserves input shape."""
    z = np.linspace(0.0, 2.0, 50)
    mus = np.array([0.4, 1.2])
    sigmas = np.array([0.2, 0.4])
    got = gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas)
    assert got.shape == z.shape


def test_gaussian_mixture_raises_for_non_1d_mus_or_sigmas():
    """Tests that gaussian_mixture_distribution raises ValueError
    for non-1D mus or sigmas."""
    z = np.linspace(0.0, 1.0, 5)
    mus = np.array([[0.0, 1.0]])
    sigmas = np.array([0.2, 0.3])
    with pytest.raises(ValueError, match="mus and sigmas must be 1D arrays"):
        gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas)


def test_gaussian_mixture_raises_for_length_mismatch():
    """Tests that gaussian_mixture_distribution raises ValueError
    for length mismatch."""
    z = np.linspace(0.0, 1.0, 5)
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.2])
    with pytest.raises(ValueError, match="same length"):
        gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas)


def test_gaussian_mixture_raises_for_nonpositive_sigma():
    """Tests that gaussian_mixture_distribution raises ValueError
    for non-positive sigma."""
    z = np.linspace(0.0, 1.0, 5)
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.2, 0.0])
    with pytest.raises(ValueError, match="sigmas must be positive"):
        gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas)


def test_gaussian_mixture_raises_for_negative_weight():
    """Tests that gaussian_mixture_distribution raises ValueError
    for negative weight."""
    z = np.linspace(0.0, 1.0, 5)
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.2, 0.3])
    w = np.array([1.0, -0.1])
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas, weights=w)


def test_gaussian_mixture_raises_for_bad_weight_shape():
    """Tests that gaussian_mixture_distribution raises ValueError
    for non-1D weights."""
    z = np.linspace(0.0, 1.0, 5)
    mus = np.array([0.0, 1.0])
    sigmas = np.array([0.2, 0.3])
    w = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="weights must be a 1D array"):
        gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas, weights=w)


def test_gamma_distribution_zero_for_negative_z():
    """Tests that gamma_distribution is zero for negative z."""
    z = np.array([-2.0, -0.1, 0.0, 0.5])
    got = gamma_distribution(z, k=2.0, theta=1.3)
    assert np.all(got[:2] == 0.0)
    assert np.all(got[2:] >= 0.0)


@pytest.mark.parametrize("k,theta", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_gamma_distribution_raises_for_nonpositive_params(k, theta):
    """Tests that gamma_distribution raises ValueError for non-positive k or theta."""
    msg = "k must be positive" if k <= 0 else "theta must be positive"
    with pytest.raises(ValueError, match=msg):
        gamma_distribution(0.1, k=k, theta=theta)


def test_gamma_distribution_matches_formula_on_positive_domain():
    """Tests that gamma_distribution matches the formula for positive z."""
    z = np.array([0.0, 0.5, 2.0])
    k, theta = 3.0, 1.2
    got = gamma_distribution(z, k=k, theta=theta)
    expected = (z ** (k - 1.0)) * np.exp(-(z / theta))
    _assert_allclose(got, expected)


def test_schechter_like_distribution_matches_formula():
    """Tests that schechter_like_distribution matches the formula."""
    z = np.linspace(0.0, 3.0, 9)
    z0, alpha = 0.8, 1.4
    got = schechter_like_distribution(z, z0=z0, alpha=alpha)
    expected = (z / z0) ** alpha * np.exp(-(z / z0))
    _assert_allclose(got, expected)


@pytest.mark.parametrize("z0", [0.0, -1.0])
def test_schechter_like_distribution_raises_for_nonpositive_z0(z0):
    """Tests that schechter_like_distribution raises ValueError for non-positive z0."""
    with pytest.raises(ValueError, match="z0 must be positive"):
        schechter_like_distribution(0.2, z0=z0, alpha=1.0)


def test_lognormal_distribution_zero_for_nonpositive_z():
    """Tests that lognormal_distribution is zero for non-positive z."""
    z = np.array([-1.0, 0.0, 0.1, 1.0])
    got = lognormal_distribution(z, mu_ln=0.0, sigma_ln=1.0)
    assert np.all(got[:2] == 0.0)
    assert np.all(got[2:] > 0.0)


@pytest.mark.parametrize("sigma_ln", [0.0, -0.5])
def test_lognormal_distribution_raises_for_nonpositive_sigma_ln(sigma_ln):
    """Tests that lognormal_distribution raises ValueError for non-positive sigma_ln."""
    with pytest.raises(ValueError, match="sigma_ln must be positive"):
        lognormal_distribution(1.0, mu_ln=0.0, sigma_ln=sigma_ln)


def test_lognormal_distribution_matches_formula_on_positive_domain():
    """Tests that lognormal_distribution matches the formula for positive z."""
    z = np.array([0.2, 1.0, 2.5])
    mu_ln, sigma_ln = -0.1, 0.7
    got = lognormal_distribution(z, mu_ln=mu_ln, sigma_ln=sigma_ln)
    expected = (1.0 / z) * np.exp(-0.5 * ((np.log(z) - mu_ln) / sigma_ln) ** 2)
    _assert_allclose(got, expected)


def test_tophat_distribution_inclusive_edges():
    """Tests that tophat_distribution includes edges."""
    z = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    got = tophat_distribution(z, zmin=0.0, zmax=1.0)
    expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    _assert_allclose(got, expected)


def test_tophat_distribution_raises_for_bad_interval():
    """Tests that tophat_distribution raises ValueError"""
    with pytest.raises(ValueError, match=r"`{0,2}zmax`{0,2} must be greater than `{0,2}zmin`{0,2}"):
        tophat_distribution([0.0, 1.0], zmin=1.0, zmax=1.0)


def test_shifted_smail_distribution_zero_below_shift():
    """Tests that shifted_smail_distribution is zero below z_shift."""
    z = np.array([0.0, 0.9, 1.0, 1.2])
    z_shift = 1.0
    got = shifted_smail_distribution(z, z0=0.5, alpha=2.0, beta=1.5, z_shift=z_shift)
    assert np.all(got[z < z_shift] == 0.0)
    assert np.all(got[z >= z_shift] >= 0.0)


def test_shifted_smail_distribution_matches_formula_above_shift():
    """Tests that shifted_smail_distribution matches the formula above z_shift."""
    z = np.array([1.0, 1.5, 2.0])
    z0, alpha, beta, z_shift = 0.4, 1.3, 2.2, 1.0
    got = shifted_smail_distribution(z, z0=z0, alpha=alpha, beta=beta, z_shift=z_shift)
    zz = (z - z_shift) / z0
    expected = (zz**alpha) * np.exp(-(zz**beta))
    _assert_allclose(got, expected)


@pytest.mark.parametrize("z0", [0.0, -0.1])
def test_shifted_smail_distribution_raises_for_nonpositive_z0(z0):
    """Tests that shifted_smail_distribution raises ValueError for non-positive z0."""
    with pytest.raises(ValueError, match="z0 must be positive"):
        shifted_smail_distribution(1.0, z0=z0, alpha=1.0, beta=1.0, z_shift=0.0)


def test_skew_normal_distribution_reduces_to_half_gaussian_when_alpha_zero():
    """Tests that skew_normal_distribution reduces to half Gaussian when alpha=0."""
    z = np.linspace(-1.0, 1.0, 11)
    xi, omega = 0.2, 0.4
    got = skew_normal_distribution(z, xi=xi, omega=omega, alpha=0.0)
    t = (z - xi) / omega
    expected = 0.5 * np.exp(-0.5 * t**2)
    _assert_allclose(got, expected)


@pytest.mark.parametrize("omega", [0.0, -1.0])
def test_skew_normal_distribution_raises_for_nonpositive_omega(omega):
    """Tests that skew_normal_distribution raises ValueError for non-positive omega."""
    with pytest.raises(ValueError, match="omega must be positive"):
        skew_normal_distribution(0.0, xi=0.0, omega=omega, alpha=1.0)


def test_skew_normal_distribution_outputs_nonnegative():
    """Tests that skew_normal_distribution outputs non-negative values."""
    z = np.linspace(-2.0, 3.0, 50)
    got = skew_normal_distribution(z, xi=0.5, omega=0.8, alpha=5.0)
    assert np.all(got >= 0.0)


def test_student_t_distribution_peak_is_one_at_mu():
    """Tests that student_t_distribution peaks at 1.0 when z=mu."""
    mu, sigma, nu = 0.3, 0.7, 4.0
    got = student_t_distribution(mu, mu=mu, sigma=sigma, nu=nu)
    _assert_allclose(got, 1.0)


def test_student_t_distribution_is_symmetric_about_mu():
    """Tests that student_t_distribution is symmetric about mu."""
    mu, sigma, nu, d = 0.0, 0.5, 3.5, 0.2
    left = student_t_distribution(mu - d, mu=mu, sigma=sigma, nu=nu)
    right = student_t_distribution(mu + d, mu=mu, sigma=sigma, nu=nu)
    _assert_allclose(left, right)


@pytest.mark.parametrize(
    "sigma,nu,msg",
    [(-1.0, 2.0, "sigma must be positive"), (1.0, 0.0, "nu must be positive")],
)
def test_student_t_distribution_raises_for_nonpositive_params(sigma, nu, msg):
    """Tests that student_t_distribution raises ValueError for
    non-positive sigma or nu."""
    with pytest.raises(ValueError, match=msg):
        student_t_distribution(0.0, mu=0.0, sigma=sigma, nu=nu)


def test_tabulated_distribution_interpolates_expected_values():
    """Tests that tabulated_distribution interpolates expected values."""
    z = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 2.0, 0.0])

    got = tabulated_distribution(z, z_input=z_input, nz_input=nz_input)

    expected = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    _assert_allclose(got, expected)


def test_tabulated_distribution_returns_zero_outside_input_range():
    """Tests that tabulated_distribution returns zero outside input range."""
    z = np.array([-0.5, 0.0, 1.0, 2.0, 2.5])
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 1.0, 0.0])

    got = tabulated_distribution(z, z_input=z_input, nz_input=nz_input)

    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    _assert_allclose(got, expected)


def test_tabulated_distribution_preserves_shape_for_array_input():
    """Tests that tabulated_distribution preserves input shape."""
    z = np.linspace(0.0, 2.0, 25)
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 1.0, 0.0])

    got = tabulated_distribution(z, z_input=z_input, nz_input=nz_input)

    assert got.shape == z.shape


def test_tabulated_distribution_scalar_input_returns_scalar_shape():
    """Tests that tabulated_distribution preserves scalar input shape."""
    z = 0.5
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 2.0, 0.0])

    got = tabulated_distribution(z, z_input=z_input, nz_input=nz_input)

    assert np.asarray(got).shape == ()
    _assert_allclose(got, 1.0)


def test_tabulated_distribution_normalizes_over_z():
    """Tests that tabulated_distribution normalizes over z."""
    z = np.linspace(0.0, 2.0, 101)
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 2.0, 0.0])

    got = tabulated_distribution(z, z_input=z_input, nz_input=nz_input, normalize=True)

    _assert_allclose(np.trapezoid(got, z), 1.0, rtol=1e-12, atol=1e-12)


def test_tabulated_distribution_raises_for_non_1d_input_arrays():
    """Tests that tabulated_distribution raises ValueError for non-1D inputs."""
    z = np.linspace(0.0, 2.0, 5)
    z_input = np.array([[0.0, 1.0, 2.0]])
    nz_input = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="z_table and nz_table must be 1D arrays"):
        tabulated_distribution(z, z_input=z_input, nz_input=nz_input)


def test_tabulated_distribution_raises_for_shape_mismatch():
    """Tests that tabulated_distribution raises ValueError for shape mismatch."""
    z = np.linspace(0.0, 2.0, 5)
    z_input = np.array([0.0, 1.0, 2.0])
    nz_input = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="z_table and nz_table must have the same shape"):
        tabulated_distribution(z, z_input=z_input, nz_input=nz_input)


def test_tabulated_distribution_raises_for_too_few_input_points():
    """Tests that tabulated_distribution raises ValueError for too few points."""
    z = np.linspace(0.0, 2.0, 5)
    z_input = np.array([0.0])
    nz_input = np.array([1.0])

    with pytest.raises(ValueError, match="at least two points"):
        tabulated_distribution(z, z_input=z_input, nz_input=nz_input)


@pytest.mark.parametrize(
    "z_input",
    [
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 2.0, 1.0]),
    ],
)
def test_tabulated_distribution_raises_for_non_increasing_z_input(z_input):
    """Tests that tabulated_distribution raises ValueError for bad z_input."""
    z = np.linspace(0.0, 2.0, 5)
    nz_input = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="z_table must be strictly increasing"):
        tabulated_distribution(z, z_input=z_input, nz_input=nz_input)
