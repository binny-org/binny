"""Tests for numerical consistency of predefined tomography."""

from __future__ import annotations

import numpy as np
import pytest

from binny.tomography import build_predefined_tomography


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("provide_parent", [False, True])
def test_predefined_tomography_preserves_expected_curves(normalize, provide_parent):
    """Tests that predefined tomography preserves the expected input distributions."""
    z = np.linspace(0.0, 3.0, 301)

    bins = {
        0: np.exp(-0.5 * ((z - 0.6) / 0.2) ** 2),
        1: np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2),
        2: np.exp(-0.5 * ((z - 1.9) / 0.4) ** 2),
    }

    parent = 2.0 * sum(bins.values()) if provide_parent else None

    result = build_predefined_tomography(
        z,
        bins,
        nz=parent,
        normalize=normalize,
        include_metadata=True,
    )

    np.testing.assert_allclose(result.z, z, rtol=0.0, atol=0.0)

    expected_parent = parent if provide_parent else sum(bins.values())
    np.testing.assert_allclose(
        result.nz,
        expected_parent,
        rtol=1e-12,
        atol=1e-12,
    )

    assert set(result.bins) == set(bins)

    for i, curve in bins.items():
        expected = curve.copy()

        if normalize:
            expected /= np.trapezoid(expected, x=z)

        np.testing.assert_allclose(
            result.bins[i],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    assert result.metadata is not None


@pytest.mark.parametrize("normalize", [False, True])
def test_predefined_tomography_preserves_bin_integrals(normalize):
    """Tests that predefined tomography preserves or normalizes bin integrals."""
    z = np.linspace(0.0, 3.0, 301)

    bins = {
        0: np.exp(-0.5 * ((z - 0.6) / 0.2) ** 2),
        1: np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2),
    }

    result = build_predefined_tomography(
        z,
        bins,
        normalize=normalize,
    )

    for i, curve in bins.items():
        expected_integral = 1.0 if normalize else np.trapezoid(curve, x=z)

        actual_integral = np.trapezoid(result.bins[i], x=result.z)

        np.testing.assert_allclose(
            actual_integral,
            expected_integral,
            rtol=1e-12,
            atol=1e-12,
        )
