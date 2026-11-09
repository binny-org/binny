"""Tests for functional tomography bin-combination filtering."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from binny.correlations.bin_combo_filter import BinComboFilter
from binny.nz_tomo._tomography_bins import TomographyBins
from binny.tomography.combinations import make_bin_combo_filter
from binny.tomography.core import Tomography, build_tomography


@pytest.fixture
def z() -> np.ndarray:
    """Return a shared redshift grid."""
    return np.linspace(0.0, 3.0, 301)


@pytest.fixture
def nz(z: np.ndarray) -> np.ndarray:
    """Return a smooth parent redshift distribution."""
    return z**2 * np.exp(-2.0 * z)


@pytest.fixture
def edges() -> np.ndarray:
    """Return three tomographic bins."""
    return np.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def tomo(
    z: np.ndarray,
    nz: np.ndarray,
    edges: np.ndarray,
) -> Tomography:
    """Build deterministic tomography."""
    return build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize=False,
    )


@pytest.fixture
def other_tomo() -> Tomography:
    """Build another tomography on a different redshift grid."""
    z = np.linspace(0.0, 3.0, 151)
    nz = z**2 * np.exp(-1.5 * z)

    return build_tomography(
        z,
        nz,
        np.array([0.0, 1.5, 3.0]),
        method="deterministic",
        normalize=False,
    )


def _legacy_tomo(tomo: Tomography) -> TomographyBins:
    """Construct the legacy container from a functional tomography."""
    return TomographyBins(
        z=tomo.z,
        nz=tomo.nz,
        spec={},
        bins=tomo.bins,
        tomo_meta=None,
        survey_meta=None,
    )


def test_single_tomography_returns_filter(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    assert isinstance(result, BinComboFilter)


def test_single_tomography_uses_same_grid(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    assert_array_equal(result.z, tomo.z)


def test_single_tomography_uses_two_slots(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    assert len(result.curves) == 2
    assert result.curves[0] is tomo.bins
    assert result.curves[1] is tomo.bins


def test_single_tomography_preserves_bin_keys(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    assert list(result.curves[0]) == list(tomo.bins)
    assert list(result.curves[1]) == list(tomo.bins)


def test_single_tomography_preserves_curves(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    for key, curve in tomo.bins.items():
        assert_allclose(result.curves[0][key], curve)
        assert_allclose(result.curves[1][key], curve)


def test_same_grid_two_tomographies(
    tomo: Tomography,
    z: np.ndarray,
    edges: np.ndarray,
) -> None:
    other = build_tomography(
        z,
        np.exp(-z),
        edges,
        method="deterministic",
        normalize=False,
    )

    result = make_bin_combo_filter(tomo, other)

    assert_array_equal(result.z, z)
    assert result.curves[0] is tomo.bins
    assert result.curves[1] is other.bins


def test_same_grid_preserves_distinct_samples(
    tomo: Tomography,
    z: np.ndarray,
    edges: np.ndarray,
) -> None:
    other = build_tomography(
        z,
        np.exp(-z),
        edges,
        method="deterministic",
        normalize=False,
    )

    result = make_bin_combo_filter(tomo, other)

    for key in tomo.bins:
        assert_allclose(result.curves[0][key], tomo.bins[key])
        assert_allclose(result.curves[1][key], other.bins[key])


def test_different_grids_use_union(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    result = make_bin_combo_filter(tomo, other_tomo)

    expected = np.union1d(tomo.z, other_tomo.z)

    assert_array_equal(result.z, expected)


def test_different_grids_preserve_bin_keys(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    result = make_bin_combo_filter(tomo, other_tomo)

    assert list(result.curves[0]) == list(tomo.bins)
    assert list(result.curves[1]) == list(other_tomo.bins)


def test_different_grids_interpolate_first_sample(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    result = make_bin_combo_filter(tomo, other_tomo)

    for key, curve in tomo.bins.items():
        expected = np.interp(
            result.z,
            tomo.z,
            curve,
            left=0.0,
            right=0.0,
        )

        assert_allclose(result.curves[0][key], expected)


def test_different_grids_interpolate_second_sample(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    result = make_bin_combo_filter(tomo, other_tomo)

    for key, curve in other_tomo.bins.items():
        expected = np.interp(
            result.z,
            other_tomo.z,
            curve,
            left=0.0,
            right=0.0,
        )

        assert_allclose(result.curves[1][key], expected)


def test_interpolation_outside_support_is_zero() -> None:
    z_a = np.linspace(0.0, 2.0, 101)
    z_b = np.linspace(1.0, 3.0, 101)

    tomo_a = build_tomography(
        z_a,
        np.exp(-z_a),
        np.array([0.0, 1.0, 2.0]),
        method="deterministic",
        normalize=False,
    )
    tomo_b = build_tomography(
        z_b,
        np.exp(-z_b),
        np.array([1.0, 2.0, 3.0]),
        method="deterministic",
        normalize=False,
    )

    result = make_bin_combo_filter(tomo_a, tomo_b)

    outside_a = result.z > z_a[-1]
    outside_b = result.z < z_b[0]

    for curve in result.curves[0].values():
        assert_array_equal(curve[outside_a], 0.0)

    for curve in result.curves[1].values():
        assert_array_equal(curve[outside_b], 0.0)


def test_identical_grid_values_do_not_require_same_object(
    tomo: Tomography,
    z: np.ndarray,
    edges: np.ndarray,
) -> None:
    other = build_tomography(
        z.copy(),
        np.exp(-z),
        edges,
        method="deterministic",
        normalize=False,
    )

    result = make_bin_combo_filter(tomo, other)

    assert result.z is tomo.z
    assert result.curves[1] is other.bins


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        (
            "pairs_all",
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ],
        ),
        (
            "pairs_upper_triangle",
            [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
        ),
        (
            "pairs_diagonal",
            [(0, 0), (1, 1), (2, 2)],
        ),
        (
            "pairs_off_diagonal",
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
        ),
    ],
)
def test_single_sample_topologies(
    tomo: Tomography,
    topology: str,
    expected: list[tuple[int, int]],
) -> None:
    result = make_bin_combo_filter(tomo)
    result.set_topology(topology)

    assert result.values() == expected


def test_cross_sample_cartesian_topology(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    result = make_bin_combo_filter(tomo, other_tomo)
    result.set_topology("pairs_cartesian")

    expected = [(i, j) for i in tomo.bins for j in other_tomo.bins]

    assert result.values() == expected


def test_select_topology_from_spec(tomo: Tomography) -> None:
    spec = {
        "topology": {
            "name": "pairs_upper_triangle",
        },
    }

    result = make_bin_combo_filter(tomo).select(spec)

    assert result.values() == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]


def test_select_with_score_relation(tomo: Tomography) -> None:
    spec = {
        "topology": {
            "name": "pairs_all",
        },
        "filters": [
            {
                "name": "score_relation",
                "score": "mean",
                "relation": "lt",
            },
        ],
    }

    result = make_bin_combo_filter(tomo).select(spec)

    assert result.values() == [
        (1, 0),
        (2, 0),
        (2, 1),
    ]


def test_select_with_overlap_filter(tomo: Tomography) -> None:
    spec = {
        "topology": {
            "name": "pairs_all",
        },
        "filters": [
            {
                "name": "overlap_fraction",
                "threshold": 0.0,
                "compare": "gt",
            },
        ],
    }

    result = make_bin_combo_filter(tomo).select(spec)

    assert result.values() == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_select_returns_same_filter(tomo: Tomography) -> None:
    result = make_bin_combo_filter(tomo)

    selected = result.select(
        {
            "topology": {
                "name": "pairs_diagonal",
            },
        }
    )

    assert selected is result


def test_matches_legacy_single_tomography(tomo: Tomography) -> None:
    legacy = _legacy_tomo(tomo)

    new = make_bin_combo_filter(tomo)
    old = legacy.make_bin_combo_filter()

    assert_array_equal(new.z, old.z)

    for new_slot, old_slot in zip(new.curves, old.curves, strict=True):
        assert list(new_slot) == list(old_slot)

        for key in new_slot:
            assert_allclose(new_slot[key], old_slot[key])


def test_matches_legacy_two_tomographies_same_grid(
    tomo: Tomography,
    z: np.ndarray,
    edges: np.ndarray,
) -> None:
    other = build_tomography(
        z,
        np.exp(-z),
        edges,
        method="deterministic",
        normalize=False,
    )

    new = make_bin_combo_filter(tomo, other)
    old = _legacy_tomo(tomo).make_bin_combo_filter(_legacy_tomo(other))

    assert_array_equal(new.z, old.z)

    for new_slot, old_slot in zip(new.curves, old.curves, strict=True):
        for key in new_slot:
            assert_allclose(new_slot[key], old_slot[key])


def test_matches_legacy_two_tomographies_different_grids(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    new = make_bin_combo_filter(tomo, other_tomo)
    old = _legacy_tomo(tomo).make_bin_combo_filter(_legacy_tomo(other_tomo))

    assert_array_equal(new.z, old.z)

    for new_slot, old_slot in zip(new.curves, old.curves, strict=True):
        assert list(new_slot) == list(old_slot)

        for key in new_slot:
            assert_allclose(new_slot[key], old_slot[key])


@pytest.mark.parametrize(
    "spec",
    [
        {"topology": {"name": "pairs_all"}},
        {"topology": {"name": "pairs_upper_triangle"}},
        {"topology": {"name": "pairs_diagonal"}},
        {
            "topology": {"name": "pairs_all"},
            "filters": [
                {
                    "name": "score_relation",
                    "score": "mean",
                    "relation": "lt",
                },
            ],
        },
        {
            "topology": {"name": "pairs_all"},
            "filters": [
                {
                    "name": "overlap_fraction",
                    "threshold": 0.0,
                    "compare": "gt",
                },
            ],
        },
    ],
)
def test_selection_matches_legacy(
    tomo: Tomography,
    spec: dict,
) -> None:
    new = make_bin_combo_filter(tomo).select(spec).values()
    old = _legacy_tomo(tomo).bin_combo_filter(spec)

    assert new == old


def test_cross_sample_selection_matches_legacy(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    spec = {
        "topology": {
            "name": "pairs_cartesian",
        },
        "filters": [
            {
                "name": "score_relation",
                "score": "mean",
                "relation": "lt",
            },
        ],
    }

    new = make_bin_combo_filter(tomo, other_tomo).select(spec).values()

    old = _legacy_tomo(tomo).bin_combo_filter(
        spec,
        _legacy_tomo(other_tomo),
    )

    assert new == old


def test_filter_does_not_modify_input_bins(
    tomo: Tomography,
    other_tomo: Tomography,
) -> None:
    original_a = {k: v.copy() for k, v in tomo.bins.items()}
    original_b = {k: v.copy() for k, v in other_tomo.bins.items()}

    result = make_bin_combo_filter(tomo, other_tomo)
    result.set_topology("pairs_cartesian")
    result.keep_if_overlap_fraction(threshold=0.0, compare="ge")

    for key in original_a:
        assert_array_equal(tomo.bins[key], original_a[key])

    for key in original_b:
        assert_array_equal(other_tomo.bins[key], original_b[key])
