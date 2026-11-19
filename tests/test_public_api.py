"""Tests for the public API and namespace organization."""

from importlib import import_module
from importlib.util import find_spec

import pytest

import binny
from binny import (
    analysis,
    axes,
    binning,
    calibration,
    correlations,
    cosmology,
    nz,
    style,
    surveys,
    tomography,
)

PRIMARY_NAMESPACES = (
    nz,
    binning,
    tomography,
    surveys,
    calibration,
    analysis,
    style,
)

PUBLIC_NAMESPACES = (
    binny,
    *PRIMARY_NAMESPACES,
    axes,
    correlations,
    cosmology,
)


@pytest.mark.parametrize("module", PUBLIC_NAMESPACES)
def test_public_exports_exist(module):
    """Tests that all declared public exports are available."""
    for name in module.__all__:
        assert hasattr(module, name), f"{module.__name__}: {name}"


@pytest.mark.parametrize("module", PUBLIC_NAMESPACES)
def test_public_exports_are_unique(module):
    """Tests that public export lists contain no duplicate names."""
    assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize("module", PRIMARY_NAMESPACES)
def test_primary_namespaces(module):
    """Tests that primary namespaces are exposed at the package root."""
    name = module.__name__.split(".")[-1]

    assert name in binny.__all__
    assert getattr(binny, name) is module


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (calibration, nz),
        (calibration, binning),
        (analysis, tomography),
    ],
)
def test_namespace_separation(first, second):
    """Tests that relocated functions are not exported from their former namespaces."""
    assert set(first.__all__).isdisjoint(second.__all__)

    for name in first.__all__:
        assert not hasattr(second, name)


@pytest.mark.parametrize(
    ("namespace", "module_name"),
    [
        (calibration, "binny.calibration.depth"),
        (calibration, "binny.calibration.photoz_fitting"),
        (calibration, "binny.calibration.psf_selection"),
        (analysis, "binny.analysis.combinations"),
        (analysis, "binny.analysis.comparison"),
        (analysis, "binny.analysis.composition"),
        (analysis, "binny.analysis.stats"),
        (style, "binny.style.colors"),
    ],
)
def test_relocated_implementation_exports(namespace, module_name):
    """Tests that relocated implementations are exposed through their new namespaces."""
    module = import_module(module_name)

    assert module.__all__

    for name in module.__all__:
        assert name in namespace.__all__
        assert getattr(namespace, name) is getattr(module, name)


@pytest.mark.parametrize(
    "module_name",
    [
        "binny.nz.calibration",
        "binny.nz.psf_selection",
        "binny.binning.photoz_fitting",
        "binny.tomography.combinations",
        "binny.tomography.comparison",
        "binny.tomography.composition",
        "binny.tomography.stats",
        "binny.utils.colors",
    ],
)
def test_legacy_modules_removed(module_name):
    """Tests that obsolete implementation modules no longer exist."""
    assert find_spec(module_name) is None


@pytest.mark.parametrize("name", correlations.__all__)
def test_correlation_reexports(name):
    """Tests that analysis exposes the original correlation utilities."""
    assert getattr(analysis, name) is getattr(correlations, name)


@pytest.mark.parametrize(
    "name",
    [name for name in axes.__all__ if name.endswith("_edges")],
)
def test_bin_edge_reexports(name):
    """Tests that binning exposes the original bin-edge generators."""
    assert getattr(binning, name) is getattr(axes, name)


def test_nz_loader_reexport():
    """Tests that the public redshift loader preserves its implementation."""
    from binny.utils.io import load_nz

    assert nz.load_nz is load_nz


def test_binning_recipe_loader_reexport():
    """Tests that the public recipe loader preserves its implementation."""
    from binny.utils.io import load_binning_recipe

    assert binning.load_binning_recipe is load_binning_recipe


def test_root_public_api():
    """Tests that the root API exposes only primary namespaces."""
    assert set(binny.__all__) == {
        "analysis",
        "binning",
        "calibration",
        "nz",
        "style",
        "surveys",
        "tomography",
    }


@pytest.mark.parametrize("module", PRIMARY_NAMESPACES)
def test_no_root_level_reexports(module):
    """Tests that individual pillar exports are not exposed at the package root."""
    for name in module.__all__:
        assert not hasattr(binny, name), f"Unexpected root-level export: {name}"


def test_tomography_public_api():
    """Tests that tomography only exposes construction and representation utilities."""
    assert set(tomography.__all__) == {
        "Tomography",
        "build_tomography",
        "build_tomography_from_spec",
        "build_predefined_tomography",
        "load_predefined_bins",
    }


def test_style_public_api():
    """Tests that style exposes only color and visualization utilities."""
    assert set(style.__all__) == {
        "cmap",
        "cmap_discrete",
        "get_colors",
        "norm",
        "palette",
        "pastelize",
    }
