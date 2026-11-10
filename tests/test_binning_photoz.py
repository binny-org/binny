"""Tests for photometric-redshift assignment probabilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import erf

from binny.binning.photoz import (
    photoz_assignment,
    photoz_assignment_probability,
)


def test_zero_scatter_is_deterministic():
    """Test that zero scatter produces deterministic bin assignment."""
    z = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    actual = photoz_assignment_probability(
        z,
        bin_min=0.5,
        bin_max=1.5,
        scatter_scale=0.0,
        mean_offset=0.0,
    )

    assert_array_equal(actual, [0.0, 1.0, 1.0, 0.0, 0.0])


def test_gaussian_probability_matches_analytic_cdf():
    """Test that Gaussian assignment matches the analytic CDF."""
    z = np.linspace(0.0, 2.0, 101)
    scatter = 0.05
    offset = 0.02
    scale = 1.03
    lower, upper = 0.5, 1.0

    actual = photoz_assignment_probability(
        z,
        lower,
        upper,
        scatter,
        offset,
        mean_scale=scale,
    )

    mu = scale * z - offset
    sigma = scatter * (1.0 + z)
    expected = 0.5 * (
        erf((upper - mu) / (np.sqrt(2.0) * sigma)) - erf((lower - mu) / (np.sqrt(2.0) * sigma))
    )

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_outlier_mixture_matches_weighted_components():
    """Test that the outlier mixture equals the weighted Gaussian components."""
    z = np.linspace(0.0, 2.0, 101)
    fraction = 0.3

    core = photoz_assignment_probability(z, 0.5, 1.0, 0.03, 0.0)
    outlier = photoz_assignment_probability(z, 0.5, 1.0, 0.2, 0.15)

    actual = photoz_assignment_probability(
        z,
        0.5,
        1.0,
        0.03,
        0.0,
        outlier_frac=fraction,
        outlier_scatter_scale=0.2,
        outlier_mean_offset=0.15,
    )

    assert_allclose(
        actual,
        (1.0 - fraction) * core + fraction * outlier,
        rtol=1e-13,
        atol=1e-13,
    )


def test_outlier_fraction_one_matches_outlier_component():
    """Test that a unit outlier fraction selects only the outlier component."""
    z = np.linspace(0.0, 2.0, 101)

    actual = photoz_assignment_probability(
        z,
        0.5,
        1.0,
        0.03,
        0.0,
        outlier_frac=1.0,
        outlier_scatter_scale=0.2,
        outlier_mean_offset=0.15,
    )

    expected = photoz_assignment_probability(z, 0.5, 1.0, 0.2, 0.15)

    assert_allclose(actual, expected)


def test_zero_outlier_fraction_does_not_require_outlier_scatter():
    """Test that disabled outliers do not require an outlier scatter scale."""
    z = np.linspace(0.0, 2.0, 101)

    actual = photoz_assignment_probability(z, 0.5, 1.0, 0.05, 0.0, outlier_frac=0.0)
    expected = photoz_assignment_probability(z, 0.5, 1.0, 0.05, 0.0)

    assert_array_equal(actual, expected)


def test_per_bin_parameters_match_individual_probabilities():
    """Test that per-bin parameters reproduce individual bin probabilities."""
    z = np.linspace(0.0, 2.0, 101)
    edges = np.array([0.0, 0.5, 1.0, 2.0])

    scatter = [0.02, 0.04, 0.06]
    offset = [0.0, 0.01, -0.02]
    fractions = [0.0, 0.1, 0.2]

    actual = photoz_assignment(
        z,
        edges,
        scatter_scale=scatter,
        mean_offset=offset,
        outlier_frac=fractions,
        outlier_scatter_scale=0.2,
    )

    expected = np.stack(
        [
            photoz_assignment_probability(
                z,
                edges[i],
                edges[i + 1],
                scatter[i],
                offset[i],
                outlier_frac=fractions[i],
                outlier_scatter_scale=0.2,
            )
            for i in range(3)
        ]
    )

    assert actual.shape == (3, z.size)
    assert_allclose(actual, expected)


def test_scalar_parameters_broadcast_to_all_bins():
    """Test that scalar photo-z parameters broadcast across all bins."""
    z = np.linspace(0.0, 2.0, 101)
    edges = np.array([0.0, 0.5, 1.0, 2.0])

    actual = photoz_assignment(
        z,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
    )

    assert actual.shape == (3, z.size)

    for i in range(3):
        expected = photoz_assignment_probability(z, edges[i], edges[i + 1], 0.05, 0.01)
        assert_allclose(actual[i], expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outlier_frac": -0.1}, "outlier_frac"),
        ({"outlier_frac": 1.1}, "outlier_frac"),
        ({"outlier_frac": 0.2}, "outlier_scatter_scale"),
        ({"scatter_scale": -0.1}, "scatter_scale"),
        ({"mean_scale": 0.0}, "mean_scale"),
        (
            {
                "outlier_frac": 0.2,
                "outlier_scatter_scale": -0.1,
            },
            "outlier_scatter_scale",
        ),
        (
            {
                "outlier_frac": 0.2,
                "outlier_scatter_scale": 0.1,
                "outlier_mean_scale": 0.0,
            },
            "outlier_mean_scale",
        ),
    ],
)
def test_invalid_parameters_raise(kwargs, message):
    """Test that invalid photo-z parameters raise informative errors."""
    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.0,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        photoz_assignment_probability(
            np.linspace(0.0, 2.0, 101),
            0.5,
            1.0,
            **params,
        )


def test_invalid_per_bin_parameter_length_raises():
    """Test that incorrect per-bin parameter lengths are rejected."""
    with pytest.raises(ValueError):
        photoz_assignment(
            np.linspace(0.0, 2.0, 101),
            np.array([0.0, 0.5, 1.0, 2.0]),
            scatter_scale=[0.02, 0.04],
        )
