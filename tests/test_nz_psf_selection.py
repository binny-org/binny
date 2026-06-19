"""Tests for PSF-dependent source-selection calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz.psf_selection import (
    calibrate_psf_depth_from_mock,
    effective_number_density_from_mock,
    resolution_factor,
    shear_selection_weight,
    weighted_nz_from_mock,
)


def test_resolution_factor_matches_expected_values() -> None:
    """Tests that the resolution factor follows the expected size scaling."""
    r_gal = np.array([0.0, 1.0, 2.0])

    result = resolution_factor(r_gal, r_psf=1.0)

    expected = np.array([0.0, 0.5, 0.8])
    np.testing.assert_allclose(result, expected)


def test_resolution_factor_rejects_negative_psf_size() -> None:
    """Tests that negative PSF sizes are rejected."""
    with pytest.raises(ValueError, match="r_psf must be >= 0"):
        resolution_factor(np.array([1.0]), r_psf=-0.1)


def test_shear_selection_weight_hard_threshold() -> None:
    """Tests that hard selection returns binary weights."""
    r_gal = np.array([0.5, 1.0, 2.0])

    weights = shear_selection_weight(
        r_gal,
        r_psf=1.0,
        r_min=0.5,
        kind="hard",
    )

    expected = np.array([0.0, 1.0, 1.0])
    np.testing.assert_allclose(weights, expected)


def test_shear_selection_weight_sigmoid_is_smooth_and_bounded() -> None:
    """Tests that sigmoid selection returns smooth bounded weights."""
    r_gal = np.array([0.3, 1.0, 3.0])

    weights = shear_selection_weight(
        r_gal,
        r_psf=1.0,
        r_min=0.5,
        kind="sigmoid",
        width=0.1,
    )

    assert np.all(weights > 0.0)
    assert np.all(weights < 1.0)
    assert np.all(np.diff(weights) > 0.0)


def test_shear_selection_weight_rejects_invalid_threshold() -> None:
    """Tests that invalid resolution thresholds are rejected."""
    with pytest.raises(ValueError, match="r_min must be between 0 and 1"):
        shear_selection_weight(np.array([1.0]), r_psf=1.0, r_min=1.5)


def test_shear_selection_weight_rejects_invalid_kind() -> None:
    """Tests that unsupported selection kinds are rejected."""
    with pytest.raises(ValueError, match="kind must be 'hard' or 'sigmoid'"):
        shear_selection_weight(np.array([1.0]), r_psf=1.0, kind="bad")  # type: ignore[arg-type]


def test_shear_selection_weight_rejects_nonpositive_sigmoid_width() -> None:
    """Tests that sigmoid selection requires a positive width."""
    with pytest.raises(ValueError, match="width must be > 0"):
        shear_selection_weight(
            np.array([1.0]),
            r_psf=1.0,
            kind="sigmoid",
            width=0.0,
        )


def test_weighted_nz_from_mock_returns_normalized_distribution() -> None:
    """Tests that weighted redshift distributions can be normalized."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 24.5, 25.0, 26.0])
    r_gal = np.array([2.0, 2.0, 2.0, 2.0])
    z_edges = np.array([0.0, 0.5, 1.0, 1.5])

    z, z_mid, nz, weights = weighted_nz_from_mock(
        z_true,
        mag,
        r_gal,
        maglim=25.5,
        r_psf=0.1,
        z_edges=z_edges,
        selection_kind="hard",
        normalize=True,
    )

    np.testing.assert_allclose(z, np.array([0.2, 0.4, 0.8]))
    np.testing.assert_allclose(weights, np.ones(3))
    assert z_mid.shape == nz.shape
    np.testing.assert_allclose(z_mid, np.array([0.25, 0.75, 1.25]))
    np.testing.assert_allclose(np.trapezoid(nz, z_mid), 1.0)


def test_weighted_nz_from_mock_returns_selected_redshifts_and_weights() -> None:
    """Tests that weighted redshift distributions return selected redshifts and weights."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 26.0, 24.5, 26.5])
    r_gal = np.array([2.0, 2.0, 2.0, 2.0])
    z_edges = np.array([0.0, 0.5, 1.0, 1.5])

    z, z_mid, nz, weights = weighted_nz_from_mock(
        z_true,
        mag,
        r_gal,
        maglim=25.0,
        r_psf=0.1,
        z_edges=z_edges,
        selection_kind="hard",
        normalize=False,
    )

    np.testing.assert_allclose(z, np.array([0.2, 0.8]))
    np.testing.assert_allclose(z_mid, np.array([0.25, 0.75, 1.25]))
    np.testing.assert_allclose(nz, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(weights, np.ones(2))


def test_weighted_nz_from_mock_respects_magnitude_limit() -> None:
    """Tests that weighted redshift distributions apply the magnitude limit."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 24.5, 25.0, 26.0])
    r_gal = np.array([2.0, 2.0, 2.0, 2.0])
    z_edges = np.array([0.0, 0.5, 1.0, 1.5])

    z, _, nz, weights = weighted_nz_from_mock(
        z_true,
        mag,
        r_gal,
        maglim=24.25,
        r_psf=0.1,
        z_edges=z_edges,
        selection_kind="hard",
        normalize=False,
    )

    np.testing.assert_allclose(z, np.array([0.2]))
    np.testing.assert_allclose(weights, np.ones(1))
    np.testing.assert_allclose(nz, np.array([1.0, 0.0, 0.0]))


def test_weighted_nz_from_mock_rejects_shape_mismatch() -> None:
    """Tests that weighted redshift distributions require matching catalog shapes."""
    with pytest.raises(ValueError, match="z_true, mag, and r_gal must have matching shapes"):
        weighted_nz_from_mock(
            np.array([0.1, 0.2]),
            np.array([24.0]),
            np.array([1.0, 1.0]),
            maglim=25.0,
            r_psf=0.7,
            z_edges=np.array([0.0, 1.0]),
        )


def test_weighted_nz_from_mock_rejects_invalid_z_edges() -> None:
    """Tests that weighted redshift distributions require valid redshift edges."""
    with pytest.raises(ValueError, match="z_edges must be"):
        weighted_nz_from_mock(
            np.array([0.1]),
            np.array([24.0]),
            np.array([1.0]),
            maglim=25.0,
            r_psf=0.7,
            z_edges=np.array([0.0]),
        )


def test_effective_number_density_from_mock_matches_weighted_count() -> None:
    """Tests that effective density equals weighted counts per arcmin squared."""
    mag = np.array([24.0, 25.0, 26.0])
    r_gal = np.array([2.0, 2.0, 2.0])

    neff = effective_number_density_from_mock(
        mag,
        r_gal,
        maglim=25.5,
        r_psf=0.1,
        area_deg2=1.0,
        selection_kind="hard",
    )

    np.testing.assert_allclose(neff, 2.0 / 3600.0)


def test_effective_number_density_from_mock_rejects_shape_mismatch() -> None:
    """Tests that effective density requires matching catalog shapes."""
    with pytest.raises(ValueError, match="mag and r_gal must have matching shapes"):
        effective_number_density_from_mock(
            np.array([24.0, 25.0]),
            np.array([1.0]),
            maglim=25.5,
            r_psf=0.7,
            area_deg2=1.0,
        )


def test_effective_number_density_from_mock_rejects_nonpositive_area() -> None:
    """Tests that effective density requires positive survey area."""
    with pytest.raises(ValueError, match="area_deg2 must be > 0"):
        effective_number_density_from_mock(
            np.array([24.0]),
            np.array([1.0]),
            maglim=25.5,
            r_psf=0.7,
            area_deg2=0.0,
        )


def test_calibrate_psf_depth_from_mock_returns_full_grid() -> None:
    """Tests that PSF-depth calibration evaluates every grid point."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 24.5, 25.0, 26.0])
    r_gal = np.array([2.0, 1.5, 1.0, 0.5])

    cal = calibrate_psf_depth_from_mock(
        z_true,
        mag,
        r_gal,
        maglims=np.array([24.5, 25.5]),
        r_psf_values=np.array([0.5, 1.0]),
        area_deg2=1.0,
        z_edges=np.array([0.0, 0.5, 1.0, 1.5]),
        selection_kind="hard",
    )

    assert cal["ok"]
    assert cal["model"] == "psf_depth_selection_from_mock"
    assert len(cal["results"]) == 4

    for result in cal["results"]:
        assert set(result) == {
            "maglim",
            "r_psf",
            "z",
            "z_mid",
            "nz",
            "weights",
            "neff_arcmin2",
        }
        assert result["z_mid"].shape == result["nz"].shape
        assert result["z"].shape == result["weights"].shape
        assert result["neff_arcmin2"] >= 0.0


def test_calibrate_psf_depth_from_mock_returns_selected_redshifts() -> None:
    """Tests that PSF-depth calibration returns selected redshifts."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 26.0, 24.5, 26.5])
    r_gal = np.array([2.0, 2.0, 2.0, 2.0])

    cal = calibrate_psf_depth_from_mock(
        z_true,
        mag,
        r_gal,
        maglims=np.array([25.0]),
        r_psf_values=np.array([0.1]),
        area_deg2=1.0,
        z_edges=np.array([0.0, 0.5, 1.0, 1.5]),
        selection_kind="hard",
        normalize_nz=False,
    )

    result = cal["results"][0]

    np.testing.assert_allclose(result["z"], np.array([0.2, 0.8]))
    np.testing.assert_allclose(result["z_mid"], np.array([0.25, 0.75, 1.25]))
    np.testing.assert_allclose(result["nz"], np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(result["weights"], np.ones(2))


def test_calibrate_psf_depth_from_mock_neff_decreases_with_larger_psf() -> None:
    """Tests that PSF-depth calibration reduces density for larger PSF sizes."""
    z_true = np.array([0.2, 0.4, 0.8, 1.2])
    mag = np.array([24.0, 24.5, 25.0, 25.5])
    r_gal = np.array([0.8, 0.8, 0.8, 0.8])

    cal = calibrate_psf_depth_from_mock(
        z_true,
        mag,
        r_gal,
        maglims=np.array([26.0]),
        r_psf_values=np.array([0.3, 1.2]),
        area_deg2=1.0,
        z_edges=np.array([0.0, 0.5, 1.0, 1.5]),
        selection_kind="sigmoid",
    )

    neff_small_psf = cal["results"][0]["neff_arcmin2"]
    neff_large_psf = cal["results"][1]["neff_arcmin2"]

    assert neff_large_psf < neff_small_psf


def test_resolution_factor_with_zero_psf_returns_one_for_positive_sizes() -> None:
    """Tests that zero PSF gives unit resolution for positive galaxy sizes."""
    r_gal = np.array([1.0, 2.0, 3.0])

    result = resolution_factor(r_gal, r_psf=0.0)

    np.testing.assert_allclose(result, np.ones_like(r_gal))


def test_weighted_nz_from_mock_ignores_invalid_catalog_entries() -> None:
    """Tests that weighted redshift distributions ignore invalid catalog entries."""
    z_true = np.array([0.2, -0.1, np.nan, 0.8])
    mag = np.array([24.0, 24.0, 24.0, np.inf])
    r_gal = np.array([2.0, 2.0, 2.0, 2.0])
    z_edges = np.array([0.0, 0.5, 1.0])

    z, _, nz, weights = weighted_nz_from_mock(
        z_true,
        mag,
        r_gal,
        maglim=25.0,
        r_psf=0.1,
        z_edges=z_edges,
        selection_kind="hard",
        normalize=False,
    )

    np.testing.assert_allclose(z, np.array([0.2]))
    np.testing.assert_allclose(weights, np.ones(1))
    np.testing.assert_allclose(nz, np.array([1.0, 0.0]))


def test_weighted_nz_from_mock_leaves_empty_distribution_unnormalized() -> None:
    """Tests that empty redshift distributions remain zero when normalized."""
    z_true = np.array([0.2, 0.4])
    mag = np.array([26.0, 27.0])
    r_gal = np.array([2.0, 2.0])
    z_edges = np.array([0.0, 0.5, 1.0])

    z, _, nz, weights = weighted_nz_from_mock(
        z_true,
        mag,
        r_gal,
        maglim=25.0,
        r_psf=0.1,
        z_edges=z_edges,
        selection_kind="hard",
        normalize=True,
    )

    assert z.size == 0
    assert weights.size == 0
    np.testing.assert_allclose(nz, np.zeros(2))


def test_effective_number_density_increases_with_fainter_depth() -> None:
    """Tests that effective density increases for a fainter magnitude limit."""
    mag = np.array([24.0, 25.0, 26.0])
    r_gal = np.array([2.0, 2.0, 2.0])

    neff_bright = effective_number_density_from_mock(
        mag,
        r_gal,
        maglim=24.5,
        r_psf=0.1,
        area_deg2=1.0,
        selection_kind="hard",
    )
    neff_faint = effective_number_density_from_mock(
        mag,
        r_gal,
        maglim=26.5,
        r_psf=0.1,
        area_deg2=1.0,
        selection_kind="hard",
    )

    assert neff_faint > neff_bright


def test_calibrate_psf_depth_from_mock_returns_selection_metadata() -> None:
    """Tests that PSF-depth calibration records selection metadata."""
    z_true = np.array([0.2])
    mag = np.array([24.0])
    r_gal = np.array([1.0])
    z_edges = np.array([0.0, 0.5])

    cal = calibrate_psf_depth_from_mock(
        z_true,
        mag,
        r_gal,
        maglims=np.array([25.0]),
        r_psf_values=np.array([0.7]),
        area_deg2=1.0,
        z_edges=z_edges,
        r_min=0.4,
        selection_kind="sigmoid",
        width=0.2,
    )

    assert cal["selection"] == {"kind": "sigmoid", "r_min": 0.4, "width": 0.2}
    np.testing.assert_allclose(cal["z_edges"], z_edges)
    np.testing.assert_allclose(cal["maglims"], np.array([25.0]))
    np.testing.assert_allclose(cal["r_psf_values"], np.array([0.7]))
