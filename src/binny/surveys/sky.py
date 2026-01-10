"""Conversion functions between different sky area units."""

from __future__ import annotations

from math import pi

__all__ = [
    "deg2_to_f_sky",
    "f_sky_to_deg2",
    "deg2_to_sr",
    "sr_to_deg2",
    "f_sky_to_sr",
    "sr_to_f_sky",
    "deg2_to_arcmin2",
    "arcmin2_to_deg2",
]

# Full sky solid angle in steradians
_FULL_SKY_SR = 4.0 * pi

# Conversion between steradians and square degrees
_DEG2_PER_SR = (180.0 / pi) ** 2

# Full sky area in deg^2 (IAU standard value)
_FULL_SKY_DEG2 = _FULL_SKY_SR * _DEG2_PER_SR


def deg2_to_f_sky(area_deg2: float) -> float:
    """Converts survey area in deg^2 to sky fraction f_sky.

    Args:
        area_deg2: Survey area in square degrees.

    Returns:
        Sky fraction f_sky in [0, 1].

    Raises:
        ValueError: If area_deg2 is negative.
    """
    area = float(area_deg2)

    if area < 0:
        raise ValueError("area_deg2 must be non-negative.")

    f_sky = area / _FULL_SKY_DEG2

    return f_sky


def f_sky_to_deg2(f_sky: float) -> float:
    """Converts sky fraction f_sky to survey area in deg^2.

    Args:
        f_sky: Sky fraction in [0, 1].

    Returns:
        Survey area in square degrees.

    Raises:
        ValueError: If f_sky is not in [0, 1].
    """
    f = float(f_sky)

    if f < 0 or f > 1:
        raise ValueError("f_sky must be in [0, 1].")

    deg2 = f * _FULL_SKY_DEG2

    return deg2


def deg2_to_sr(area_deg2: float) -> float:
    """Converts survey area in deg^2 to steradians.

    Args:
        area_deg2: Survey area in square degrees.

    Returns:
        Survey area in steradians.

    Raises:
        ValueError: If area_deg2 is negative.
    """
    area = float(area_deg2)

    if area < 0:
        raise ValueError("area_deg2 must be non-negative.")

    sr = area / _DEG2_PER_SR

    return sr


def sr_to_deg2(area_sr: float) -> float:
    """Converts survey area in steradians to deg^2.

    Args:
        area_sr: Survey area in steradians.

    Returns:
        Survey area in square degrees.

    Raises:
        ValueError: If area_sr is negative.
    """
    area = float(area_sr)

    if area < 0:
        raise ValueError("area_sr must be non-negative.")

    deg2 = area * _DEG2_PER_SR

    return deg2


def f_sky_to_sr(f_sky: float) -> float:
    """Converts sky fraction f_sky to survey area in steradians.

    Args:
        f_sky: Sky fraction in [0, 1].

    Returns:
        Survey area in steradians.

    Raises:
        ValueError: If f_sky is not in [0, 1].
    """
    f = float(f_sky)

    if f < 0 or f > 1:
        raise ValueError("f_sky must be in [0, 1].")

    sr = f * _FULL_SKY_SR

    return sr


def sr_to_f_sky(area_sr: float) -> float:
    """Converts survey area in steradians to sky fraction f_sky.

    Args:
        area_sr: Survey area in steradians.

    Returns:
        Sky fraction f_sky in [0, 1].

    Raises:
        ValueError: If area_sr is negative.
    """
    area = float(area_sr)

    if area < 0:
        raise ValueError("area_sr must be non-negative.")

    f_sky = area / _FULL_SKY_SR

    return f_sky


def deg2_to_arcmin2(area_deg2: float) -> float:
    """Converts survey area in deg^2 to arcmin^2.

    Args:
        area_deg2: Survey area in square degrees.

    Returns:
        Survey area in square arcminutes.

    Raises:
        ValueError: If area_deg2 is negative.
    """
    area = float(area_deg2)

    if area < 0:
        raise ValueError("area_deg2 must be non-negative.")

    arcmin2 = area * 60.0 * 60.0

    return arcmin2


def arcmin2_to_deg2(area_arcmin2: float) -> float:
    """Converts survey area in arcmin^2 to deg^2.

    Args:
        area_arcmin2: Survey area in square arcminutes.

    Returns:
        Survey area in square degrees.

    Raises:
        ValueError: If area_arcmin2 is negative.
    """
    area = float(area_arcmin2)

    if area < 0:
        raise ValueError("area_arcmin2 must be non-negative.")

    deg2 = area / (60.0 * 60.0)

    return deg2
