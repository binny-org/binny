from __future__ import annotations

import inspect

import numpy as np
import pytest

from binny.api.distributions import (
    available_redshift_distributions,
    redshift_distribution,
)
from binny.ztomo.distributions import gaussian_distribution


def _gaussian_kwargs() -> dict[str, float]:
    """Builds kwargs for gaussian_distribution based on its parameter names."""
    sig = inspect.signature(gaussian_distribution)
    params = set(sig.parameters.keys())
    params.discard("z")

    # common naming patterns
    if {"mu", "sigma"} <= params:
        return {"mu": 1.0, "sigma": 0.2}
    if {"mean", "sigma"} <= params:
        return {"mean": 1.0, "sigma": 0.2}
    if {"mean", "std"} <= params:
        return {"mean": 1.0, "std": 0.2}
    if {"loc", "scale"} <= params:
        return {"loc": 1.0, "scale": 0.2}
    if {"m", "s"} <= params:
        return {"m": 1.0, "s": 0.2}

    raise RuntimeError(
        "Test helper couldn't infer gaussian_distribution parameter names. "
        f"Signature parameters: {sorted(params)}"
    )


def test_available_redshift_distributions_sorted() -> None:
    """Tests that available_redshift_distributions returns a sorted list."""
    names = available_redshift_distributions()
    assert names == sorted(names)


def test_redshift_distribution_unknown_name_raises_and_lists_available() -> None:
    """Tests that redshift_distribution raises on unknown distribution name."""
    z = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError) as exc:
        redshift_distribution("not_a_real_dist", z)
    msg = str(exc.value)
    assert "Unknown redshift distribution" in msg
    assert "Available:" in msg
    assert "gaussian" in msg  # spot-check


def test_redshift_distribution_is_case_insensitive() -> None:
    """Tests that redshift_distribution is case-insensitive."""
    z = np.linspace(0.0, 2.0, 6)
    kwargs = _gaussian_kwargs()
    y1 = redshift_distribution("gaussian", z, **kwargs)
    y2 = redshift_distribution("GAUSSIAN", z, **kwargs)
    np.testing.assert_allclose(y1, y2)


def test_redshift_distribution_casts_z_to_float64_and_preserves_shape() -> None:
    """Tests that redshift_distribution casts z to float64 and preserves shape."""
    z32 = np.linspace(0.0, 2.0, 6, dtype=np.float32)
    kwargs = _gaussian_kwargs()
    y = redshift_distribution("gaussian", z32, **kwargs)

    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float64
    assert y.shape == z32.shape
    assert np.all(np.isfinite(y))
