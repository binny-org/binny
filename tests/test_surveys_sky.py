"""Unit tests for ``surveys.sky`` module."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from binny.surveys.sky import (
    arcmin2_to_deg2,
    arcmin2_to_sr,
    area_to_arcmin2,
    deg2_to_arcmin2,
    deg2_to_f_sky,
    deg2_to_sr,
    density_arcmin2_to_sr,
    density_sr_to_arcmin2,
    density_to_per_arcmin2,
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


def test_area_to_arcmin2_rejects_unknown_unit() -> None:
    """Tests that area_to_arcmin2 rejects unknown unit strings."""
    with pytest.raises(ValueError, match="unit must be one of"):
        area_to_arcmin2(1.0, unit="m2")


def test_area_to_arcmin2_rejects_negative_area() -> None:
    """Tests that area_to_arcmin2 rejects negative area values."""
    with pytest.raises(ValueError, match="area must be non-negative"):
        area_to_arcmin2(-1.0, unit="deg2")


def test_area_to_arcmin2_warns_on_zero_area() -> None:
    """Tests that area_to_arcmin2 warns when area is 0."""
    with pytest.warns(UserWarning, match="area is 0"):
        out = area_to_arcmin2(0.0, unit="deg2")
    assert out == 0.0


@pytest.mark.parametrize("unit", ["arcmin2", "arcmin^2"])
def test_area_to_arcmin2_identity_for_arcmin2_units(unit: str) -> None:
    """Tests that area_to_arcmin2 is identity for arcmin2-like units."""
    out = area_to_arcmin2(12.5, unit=unit)
    assert np.isclose(out, 12.5, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("unit", ["deg2", "deg^2"])
def test_area_to_arcmin2_converts_from_deg2(unit: str) -> None:
    """Tests that area_to_arcmin2 converts from deg2 to arcmin2."""
    out = area_to_arcmin2(2.0, unit=unit)
    assert np.isclose(out, deg2_to_arcmin2(2.0), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("unit", ["sr", "steradian", "steradians"])
def test_area_to_arcmin2_converts_from_sr(unit: str) -> None:
    """Tests that area_to_arcmin2 converts from sr to arcmin2."""
    out = area_to_arcmin2(1.23, unit=unit)
    assert np.isclose(out, sr_to_arcmin2(1.23), rtol=0.0, atol=1e-12)


def test_density_to_per_arcmin2_rejects_unknown_unit() -> None:
    """Tests that density_to_per_arcmin2 rejects unknown unit strings."""
    with pytest.raises(ValueError, match="unit must be one of"):
        density_to_per_arcmin2(1.0, unit="m2")


def test_density_to_per_arcmin2_rejects_negative_density() -> None:
    """Tests that density_to_per_arcmin2 rejects negative density values."""
    with pytest.raises(ValueError, match="density must be non-negative"):
        density_to_per_arcmin2(-1.0, unit="arcmin2")


def test_density_to_per_arcmin2_warns_on_zero_density() -> None:
    """Tests that density_to_per_arcmin2 warns when density is 0."""
    with pytest.warns(UserWarning, match="density is 0"):
        out = density_to_per_arcmin2(0.0, unit="deg2")
    assert out == 0.0


@pytest.mark.parametrize("unit", ["arcmin2", "arcmin^2"])
def test_density_to_per_arcmin2_identity_for_arcmin2_units(unit: str) -> None:
    """Tests that density_to_per_arcmin2 is identity for arcmin2-like units."""
    out = density_to_per_arcmin2(3.4, unit=unit)
    assert np.isclose(out, 3.4, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("unit", ["deg2", "deg^2"])
def test_density_to_per_arcmin2_converts_from_deg2(unit: str) -> None:
    """Tests that density_to_per_arcmin2 converts from per-deg2 to per-arcmin2."""
    d = 3600.0  # 1 gal/arcmin2 expressed as gal/deg2
    out = density_to_per_arcmin2(d, unit=unit)
    assert np.isclose(out, 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("unit", ["sr", "steradian", "steradians"])
def test_density_to_per_arcmin2_converts_from_sr(unit: str) -> None:
    """Tests that density_to_per_arcmin2 converts from per-sr to per-arcmin2."""
    d_sr = 123.0
    out = density_to_per_arcmin2(d_sr, unit=unit)
    expected = d_sr / float(sr_to_arcmin2(1.0))
    assert np.isclose(out, expected, rtol=0.0, atol=1e-12)
