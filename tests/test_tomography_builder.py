"""Tests for the functional tomography specification builder."""

import numpy as np
import pytest

from binny.binning.photoz import build_photoz_bins
from binny.binning.specz import build_specz_bins
from binny.tomography.builders import build_tomography_from_spec
from binny.tomography.predefined import build_predefined_tomography


@pytest.fixture
def sample():
    """Return a synthetic parent redshift distribution."""
    z = np.linspace(0.0, 3.0, 301)
    nz = z**2 * np.exp(-z / 0.5)
    return z, nz


@pytest.mark.parametrize("kind", ["photoz", "specz"])
@pytest.mark.parametrize("include_metadata", [False, True])
def test_matches_numerical_builder(sample, kind, include_metadata):
    """Tests that the high-level builder reproduces the numerical builder."""
    z, nz = sample

    edges = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]

    spec = {
        "kind": kind,
        "bins": {"edges": edges},
        "uncertainties": {},
    }

    result = build_tomography_from_spec(
        z=z,
        nz=nz,
        tomo_spec=spec,
        include_tomo_metadata=include_metadata,
    )

    builder = build_photoz_bins if kind == "photoz" else build_specz_bins

    expected = builder(
        z=z,
        nz=nz,
        bin_edges=edges,
        binning_scheme=None,
        n_bins=None,
        include_metadata=include_metadata,
    )

    if include_metadata:
        expected_bins, _ = expected
    else:
        expected_bins = expected

    np.testing.assert_allclose(result.z, z)
    np.testing.assert_allclose(result.nz, nz)

    assert result.bins.keys() == expected_bins.keys()

    for index in expected_bins:
        np.testing.assert_allclose(
            result.bins[index],
            expected_bins[index],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("kind", ["photoz", "specz"])
def test_missing_arrays(kind):
    """Tests that missing parent arrays raise an error."""
    spec = {
        "kind": kind,
        "bins": {"edges": [0.0, 1.0, 2.0]},
    }

    with pytest.raises(ValueError, match="z and nz are required"):
        build_tomography_from_spec(tomo_spec=spec)


@pytest.mark.parametrize("normalize", [False, True])
def test_predefined_bins(sample, normalize):
    """Tests that predefined bins match the functional predefined builder."""
    z, _ = sample

    bins = {
        0: np.exp(-0.5 * ((z - 0.6) / 0.2) ** 2),
        1: np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2),
    }

    spec = {
        "kind": "predefined",
        "bins": {
            "scheme": "predefined",
            "source": {"path": "unused.txt"},
            "normalize_bins": normalize,
        },
    }

    result = build_tomography_from_spec(
        z=z,
        tomo_spec=spec,
        predefined_bins=bins,
    )

    expected = build_predefined_tomography(
        z,
        bins,
        normalize=normalize,
    )

    np.testing.assert_allclose(result.z, expected.z)
    np.testing.assert_allclose(result.nz, expected.nz)

    assert result.bins.keys() == expected.bins.keys()

    for index in expected.bins:
        np.testing.assert_allclose(
            result.bins[index],
            expected.bins[index],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("kind", ["photoz", "specz"])
def test_metadata_option(sample, kind):
    """Tests that numerical metadata is included only when requested."""
    z, nz = sample

    spec = {
        "kind": kind,
        "bins": {"edges": [0.0, 1.0, 2.0, 3.0]},
        "uncertainties": {},
    }

    without_metadata = build_tomography_from_spec(
        z=z,
        nz=nz,
        tomo_spec=spec,
    )

    with_metadata = build_tomography_from_spec(
        z=z,
        nz=nz,
        tomo_spec=spec,
        include_tomo_metadata=True,
    )

    assert without_metadata.metadata == {}
    assert with_metadata.metadata
    assert without_metadata.spec is not None
    assert without_metadata.spec["kind"] == kind


@pytest.mark.parametrize(
    ("include_metadata", "include_diagnostics"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_photoz_diagnostics_options(sample, include_metadata, include_diagnostics):
    """Tests that photo-z metadata and diagnostics can be requested independently."""
    z, nz = sample

    spec = {
        "kind": "photoz",
        "bins": {"edges": [0.0, 1.0, 2.0, 3.0]},
        "uncertainties": {},
    }

    result = build_tomography_from_spec(
        z=z,
        nz=nz,
        tomo_spec=spec,
        include_tomo_metadata=include_metadata,
        include_photoz_diagnostics=include_diagnostics,
    )

    assert isinstance(result.metadata, dict)
    assert result.spec is not None
    assert result.spec["kind"] == "photoz"
    assert len(result.bins) == 3

    if not include_metadata and not include_diagnostics:
        assert result.metadata == {}
    else:
        assert result.metadata


@pytest.mark.parametrize("kind", ["photoz", "specz"])
def test_missing_bins_specification(sample, kind):
    """Tests that tomography requires a valid bins specification."""
    z, nz = sample

    with pytest.raises((ValueError, KeyError, TypeError)):
        build_tomography_from_spec(
            z=z,
            nz=nz,
            tomo_spec={"kind": kind},
        )
