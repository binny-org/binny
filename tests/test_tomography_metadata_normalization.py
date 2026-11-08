"""Unit tests for tomographic population-fraction metadata."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.photoz import build_photoz_bins
from binny.nz_tomo.specz import build_specz_bins


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (build_photoz_bins, {"scatter_scale": 0.05}),
        (build_specz_bins, {}),
    ],
    ids=["photoz", "specz"],
)
@pytest.mark.parametrize("normalize_bins", [False, True])
def test_metadata_normalization_consistency(
    builder,
    kwargs,
    normalize_bins,
):
    """Tests that parent normalization and population fractions are consistent."""
    z = np.linspace(0.0, 2.0, 1001)

    # Deliberately unnormalized parent distribution.
    nz = 7.0 * z**2 * np.exp(-3.0 * z)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    results = {}

    for normalize_input in (False, True):
        _, meta = builder(
            z=z,
            nz=nz,
            bin_edges=edges,
            normalize_input=normalize_input,
            normalize_bins=normalize_bins,
            include_metadata=True,
            **kwargs,
        )

        parent = meta["parent_nz"]
        bins_meta = meta["bins"]

        # The stored parent norm must match the stored parent curve.
        np.testing.assert_allclose(
            parent["norm"],
            np.trapezoid(parent["values"], x=z),
            rtol=1e-12,
            atol=1e-12,
        )

        # Population fractions must be computed from raw bin integrals.
        for i, fraction in bins_meta["frac_per_bin"].items():
            np.testing.assert_allclose(
                fraction,
                bins_meta["bins_norms"][i] / parent["norm"],
                rtol=1e-12,
                atol=1e-12,
            )

        results[normalize_input] = bins_meta["frac_per_bin"]

    # Normalizing the input must not change physical bin populations.
    for i in results[False]:
        np.testing.assert_allclose(
            results[False][i],
            results[True][i],
            rtol=1e-12,
            atol=1e-12,
        )
