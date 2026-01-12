"""Unit tests for ``surveys.sky`` module."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from binny.surveys.sky import (
    arcmin2_to_deg2,
    arcmin2_to_sr,
    deg2_to_arcmin2,
    deg2_to_f_sky,
    deg2_to_sr,
    density_arcmin2_to_sr,
    density_sr_to_arcmin2,
    f_sky_to_deg2,
    f_sky_to_sr,
    sr_to_arcmin2,
    sr_to_deg2,
    sr_to_f_sky,
)

# These are duplicated here intentionally as test reference values.
_FULL_SKY_SR = 4.0 * pi
_DEG2_PER_SR = (180.0 / pi) ** 2
_FULL_SKY_DEG2 = _FULL_SKY_SR * _DEG2_PER_SR


@pytest.mark.parametrize("area_deg2", [0.0, 1.0, 12.34, 1e4, _FULL_SKY_DEG2])
def test_deg2_sr_roundtrip(area_deg2: float) -> None:
    """Tests that deg2_to_sr and sr_to_deg2 are inverses."""
    sr = deg2_to_sr(area_deg2)
    back = sr_to_deg2(sr)
    assert np.isclose(back, area_deg2, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("area_deg2", [0.0, 1.0, 12.34, 1e4])
def test_deg2_arcmin2_roundtrip(area_deg2: float) -> None:
    """Tests that deg2_to_arcmin2 and arcmin2_to_deg2 are inverses."""
    a2 = deg2_to_arcmin2(area_deg2)
    back = arcmin2_to_deg2(a2)
    assert np.isclose(back, area_deg2, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("area_sr", [0.0, 1.0, 3.21, 10.0, _FULL_SKY_SR])
def test_sr_arcmin2_roundtrip(area_sr: float) -> None:
    """Tests that sr_to_arcmin2 and arcmin2_to_sr are inverses."""
    a2 = sr_to_arcmin2(area_sr)
    back = arcmin2_to_sr(a2)
    assert np.isclose(back, area_sr, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("f_sky", [0.0, 0.1, 0.5, 1.0])
def test_fsky_deg2_roundtrip(f_sky: float) -> None:
    """Tests that f_sky_to_deg2 and deg2_to_f_sky are inverses."""
    deg2 = f_sky_to_deg2(f_sky)
    back = deg2_to_f_sky(deg2)
    assert np.isclose(back, f_sky, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("f_sky", [0.0, 0.1, 0.5, 1.0])
def test_fsky_sr_roundtrip(f_sky: float) -> None:
    """Tests that f_sky_to_sr and sr_to_f_sky are inverses."""
    sr = f_sky_to_sr(f_sky)
    back = sr_to_f_sky(sr)
    assert np.isclose(back, f_sky, rtol=0.0, atol=1e-12)


def test_full_sky_consistency() -> None:
    """Tests that f_sky_to_sr and f_sky_to_deg2 return the full sky area."""
    assert np.isclose(f_sky_to_sr(1.0), _FULL_SKY_SR, rtol=0.0, atol=1e-12)
    assert np.isclose(f_sky_to_deg2(1.0), _FULL_SKY_DEG2, rtol=0.0, atol=1e-10)

    # And the inverse maps should return 1.
    assert np.isclose(sr_to_f_sky(_FULL_SKY_SR), 1.0, rtol=0.0, atol=1e-12)
    assert np.isclose(deg2_to_f_sky(_FULL_SKY_DEG2), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "fn,arg",
    [
        (deg2_to_f_sky, -1.0),
        (deg2_to_sr, -1.0),
        (sr_to_deg2, -1.0),
        (deg2_to_arcmin2, -1.0),
        (arcmin2_to_deg2, -1.0),
        (arcmin2_to_sr, -1.0),
        (sr_to_arcmin2, -1.0),
        (sr_to_f_sky, -1.0),
    ],
)
def test_negative_inputs_raise(fn, arg: float) -> None:
    """Tests that negative inputs raise ValueError."""
    with pytest.raises(ValueError):
        fn(arg)


@pytest.mark.parametrize("f_sky", [-1e-6, -0.1, 1.0 + 1e-12, 1.1, 2.0])
def test_f_sky_to_deg2_out_of_range_raises(f_sky: float) -> None:
    """Tests that f_sky_to_deg2 raises ValueError for out-of-range inputs."""
    with pytest.raises(ValueError):
        f_sky_to_deg2(f_sky)


@pytest.mark.parametrize("f_sky", [-1e-6, -0.1, 1.0 + 1e-12, 1.1, 2.0])
def test_f_sky_to_sr_out_of_range_raises(f_sky: float) -> None:
    """Tests that f_sky_to_sr raises ValueError for out-of-range inputs."""
    with pytest.raises(ValueError):
        f_sky_to_sr(f_sky)


def test_density_arcmin2_to_sr_and_back_roundtrip() -> None:
    """Tests that density_arcmin2_to_sr and density_sr_to_arcmin2 are inverses."""
    dens_arcmin2 = {0: 10.0, 1: 20.5, 2: 0.0}
    dens_sr = density_arcmin2_to_sr(dens_arcmin2)
    back = density_sr_to_arcmin2(dens_sr)

    for k, v in dens_arcmin2.items():
        assert np.isclose(back[int(k)], float(v), rtol=0.0, atol=1e-12)


def test_density_conversions_cast_keys_to_int() -> None:
    """Tests that density_arcmin2_to_sr and density_sr_to_arcmin2 cast keys to int."""
    dens_arcmin2 = {0.0: 10.0, 1.0: 12.0}
    dens_sr = density_arcmin2_to_sr(dens_arcmin2)
    assert set(dens_sr.keys()) == {0, 1}


@pytest.mark.parametrize(
    "dens",
    [
        {0: -1.0},
        {0: 1.0, 2: -0.5},
    ],
)
def test_density_arcmin2_to_sr_negative_raises(dens) -> None:
    """Tests that density_arcmin2_to_sr raises ValueError for negative densities."""
    with pytest.raises(ValueError):
        density_arcmin2_to_sr(dens)


@pytest.mark.parametrize(
    "dens",
    [
        {0: -1.0},
        {0: 1.0, 2: -0.5},
    ],
)
def test_density_sr_to_arcmin2_negative_raises(dens) -> None:
    """Tests that density_sr_to_arcmin2 raises ValueError for negative densities."""
    with pytest.raises(ValueError):
        density_sr_to_arcmin2(dens)


def test_arcmin2_per_sr_factor_is_consistent() -> None:
    """Tests that arcmin2_per_sr is consistent with the expected value."""
    arcmin2_per_sr = sr_to_arcmin2(1.0)
    expected = _DEG2_PER_SR * 60.0 * 60.0
    assert np.isclose(arcmin2_per_sr, expected, rtol=0.0, atol=1e-10)
