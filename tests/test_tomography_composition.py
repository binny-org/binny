"""Tests for functional tomography sample composition."""

from __future__ import annotations

import numpy as np
import pytest

from binny.tomography.composition import (
    align_bin_grids,
    combine_parent_nz,
    combine_tomographies,
    interpolate_tomography,
    sample_bin_labels,
    sample_bins,
    sample_combinations,
)
from binny.tomography.core import Tomography


def make_tomo(
    *,
    z: np.ndarray,
    nz: np.ndarray,
    bins: dict[int, np.ndarray],
) -> Tomography:
    """Tests use a small tomography object."""
    return Tomography(
        z=z,
        nz=nz,
        bins=bins,
        metadata={"kind": "test", "survey": "mock"},
    )


def test_combine_parent_nz_sums_survey_like_samples() -> None:
    """Tests that parent n(z) curves from multiple samples are summed."""
    z = np.linspace(0.0, 2.0, 5)

    bgs = {"z": z, "nz": np.array([0.0, 1.0, 2.0, 1.0, 0.0])}
    lrg = {"z": z, "nz": np.array([0.0, 0.5, 1.5, 2.0, 0.5])}

    z_out, nz_out = combine_parent_nz([bgs, lrg])

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(nz_out, bgs["nz"] + lrg["nz"])


def test_combine_parent_nz_interpolates_to_first_grid() -> None:
    """Tests that parent samples can be interpolated onto the first grid."""
    z0 = np.array([0.0, 0.5, 1.0])
    z1 = np.array([0.0, 1.0])

    sample_a = {"z": z0, "nz": np.array([0.0, 1.0, 0.0])}
    sample_b = {"z": z1, "nz": np.array([0.0, 2.0])}

    z_out, nz_out = combine_parent_nz([sample_a, sample_b], interpolate=True)

    np.testing.assert_allclose(z_out, z0)
    np.testing.assert_allclose(nz_out, np.array([0.0, 2.0, 2.0]))


def test_combine_parent_nz_interpolates_to_target_grid() -> None:
    """Tests that parent samples can be interpolated onto a target grid."""
    z_target = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    samples = [
        {"z": np.array([0.0, 1.0]), "nz": np.array([0.0, 2.0])},
        {"z": np.array([0.0, 0.5, 1.0]), "nz": np.array([0.0, 1.0, 0.0])},
    ]

    z_out, nz_out = combine_parent_nz(
        samples,
        interpolate=True,
        z_target=z_target,
    )

    expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    expected += np.array([0.0, 0.5, 1.0, 0.5, 0.0])

    np.testing.assert_allclose(z_out, z_target)
    np.testing.assert_allclose(nz_out, expected)


def test_combine_parent_nz_rejects_mismatched_redshift_grid() -> None:
    """Tests that parent samples must share one redshift grid."""
    samples = [
        {"z": np.array([0.0, 0.5, 1.0]), "nz": np.ones(3)},
        {"z": np.array([0.0, 0.6, 1.0]), "nz": np.ones(3)},
    ]

    with pytest.raises(ValueError, match="same z grid"):
        combine_parent_nz(samples)


def test_combine_parent_nz_rejects_empty_input() -> None:
    """Tests that combining parent samples requires at least one sample."""
    with pytest.raises(ValueError, match="At least one sample"):
        combine_parent_nz([])


def test_interpolate_tomography_maps_parent_and_bins() -> None:
    """Tests that a tomography object is interpolated onto a target grid."""
    z = np.array([0.0, 1.0])
    z_target = np.array([0.0, 0.5, 1.0, 1.5])

    sample = make_tomo(
        z=z,
        nz=np.array([0.0, 2.0]),
        bins={0: np.array([0.0, 1.0]), 1: np.array([2.0, 0.0])},
    )

    out = interpolate_tomography(sample, z_target)

    np.testing.assert_allclose(out.z, z_target)
    np.testing.assert_allclose(out.nz, np.array([0.0, 1.0, 2.0, 0.0]))
    np.testing.assert_allclose(out.bins[0], np.array([0.0, 0.5, 1.0, 0.0]))
    np.testing.assert_allclose(out.bins[1], np.array([2.0, 1.0, 0.0, 0.0]))

    assert out.metadata == sample.metadata
    assert out is not sample


def test_combine_tomographies_sums_matching_bins_and_parent_nz() -> None:
    """Tests that matching tomographic bins are summed bin-by-bin."""
    z = np.linspace(0.0, 2.0, 5)

    bgs = make_tomo(
        z=z,
        nz=np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        bins={
            0: np.array([0.0, 1.0, 0.5, 0.0, 0.0]),
            1: np.array([0.0, 0.0, 1.5, 1.0, 0.0]),
        },
    )
    lrg = make_tomo(
        z=z,
        nz=np.array([0.0, 0.5, 1.5, 2.0, 0.5]),
        bins={
            0: np.array([0.0, 0.5, 0.5, 0.0, 0.0]),
            1: np.array([0.0, 0.0, 1.0, 2.0, 0.5]),
        },
    )

    combined = combine_tomographies([bgs, lrg])

    np.testing.assert_allclose(combined.z, z)
    np.testing.assert_allclose(combined.nz, bgs.nz + lrg.nz)
    np.testing.assert_allclose(combined.bins[0], bgs.bins[0] + lrg.bins[0])
    np.testing.assert_allclose(combined.bins[1], bgs.bins[1] + lrg.bins[1])

    assert combined.metadata == {"kind": "combined"}


def test_combine_tomographies_interpolates_to_first_grid() -> None:
    """Tests that matching tomographic bins can be interpolated and summed."""
    z0 = np.array([0.0, 0.5, 1.0])
    z1 = np.array([0.0, 1.0])

    sample_a = make_tomo(
        z=z0,
        nz=np.array([0.0, 1.0, 0.0]),
        bins={0: np.array([0.0, 1.0, 0.0])},
    )
    sample_b = make_tomo(
        z=z1,
        nz=np.array([0.0, 2.0]),
        bins={0: np.array([0.0, 4.0])},
    )

    combined = combine_tomographies([sample_a, sample_b], interpolate=True)

    np.testing.assert_allclose(combined.z, z0)
    np.testing.assert_allclose(combined.nz, np.array([0.0, 2.0, 2.0]))
    np.testing.assert_allclose(combined.bins[0], np.array([0.0, 3.0, 4.0]))


def test_combine_tomographies_interpolates_to_target_grid() -> None:
    """Tests that matching tomographic bins can use a requested target grid."""
    z_target = np.array([0.0, 0.5, 1.0])

    sample_a = make_tomo(
        z=np.array([0.0, 1.0]),
        nz=np.array([0.0, 2.0]),
        bins={0: np.array([0.0, 4.0])},
    )
    sample_b = make_tomo(
        z=np.array([0.0, 0.5, 1.0]),
        nz=np.array([0.0, 1.0, 0.0]),
        bins={0: np.array([0.0, 2.0, 0.0])},
    )

    combined = combine_tomographies(
        [sample_a, sample_b],
        interpolate=True,
        z_target=z_target,
    )

    np.testing.assert_allclose(combined.z, z_target)
    np.testing.assert_allclose(combined.nz, np.array([0.0, 2.0, 2.0]))
    np.testing.assert_allclose(combined.bins[0], np.array([0.0, 4.0, 4.0]))


def test_combine_tomographies_rejects_mismatched_redshift_grid() -> None:
    """Tests that tomographic samples must share one redshift grid."""
    sample_a = make_tomo(
        z=np.array([0.0, 0.5, 1.0]),
        nz=np.ones(3),
        bins={0: np.ones(3)},
    )
    sample_b = make_tomo(
        z=np.array([0.0, 0.6, 1.0]),
        nz=np.ones(3),
        bins={0: np.ones(3)},
    )

    with pytest.raises(ValueError, match="same z grid"):
        combine_tomographies([sample_a, sample_b])


def test_combine_tomographies_rejects_mismatched_bin_keys() -> None:
    """Tests that tomographic samples must have the same bin labels."""
    z = np.linspace(0.0, 1.0, 3)

    sample_a = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={0: np.ones(3), 1: np.ones(3)},
    )
    sample_b = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={0: np.ones(3), 2: np.ones(3)},
    )

    with pytest.raises(ValueError, match="same bin keys"):
        combine_tomographies([sample_a, sample_b])


def test_combine_tomographies_rejects_empty_input() -> None:
    """Tests that combining tomography objects requires at least one sample."""
    with pytest.raises(ValueError, match="At least one tomography"):
        combine_tomographies([])


def test_sample_bin_labels_preserve_sample_identity() -> None:
    """Tests that equal bin indices from different samples remain distinct."""
    z = np.linspace(0.0, 1.0, 3)

    samples = {
        "BGS": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
        "LRG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3), 1: np.ones(3)}),
    }

    assert sample_bin_labels(samples) == [
        ("BGS", 0),
        ("LRG", 0),
        ("LRG", 1),
    ]


def test_sample_bins_returns_flat_sample_aware_bin_dictionary() -> None:
    """Tests that flattened bins are keyed by sample name and bin index."""
    z = np.linspace(0.0, 1.0, 3)

    bgs_bin = np.array([0.0, 1.0, 0.0])
    lrg_bin = np.array([0.0, 0.5, 0.5])

    samples = {
        "BGS": make_tomo(z=z, nz=np.ones(3), bins={0: bgs_bin}),
        "LRG": make_tomo(z=z, nz=np.ones(3), bins={0: lrg_bin}),
    }

    bins = sample_bins(samples)

    assert list(bins) == [("BGS", 0), ("LRG", 0)]
    np.testing.assert_allclose(bins[("BGS", 0)], bgs_bin)
    np.testing.assert_allclose(bins[("LRG", 0)], lrg_bin)


def test_sample_combinations_builds_lens_source_pairs() -> None:
    """Tests that sample-aware lens-source pair labels are generated."""
    z = np.linspace(0.0, 2.0, 5)

    lenses = {
        "BGS": make_tomo(z=z, nz=np.ones(5), bins={0: np.ones(5)}),
        "LRG": make_tomo(z=z, nz=np.ones(5), bins={0: np.ones(5), 1: np.ones(5)}),
    }
    sources = {
        "LSST": make_tomo(z=z, nz=np.ones(5), bins={0: np.ones(5), 1: np.ones(5)}),
    }

    pairs = sample_combinations(lenses, sources)

    assert pairs == [
        (("BGS", 0), ("LSST", 0)),
        (("BGS", 0), ("LSST", 1)),
        (("LRG", 0), ("LSST", 0)),
        (("LRG", 0), ("LSST", 1)),
        (("LRG", 1), ("LSST", 0)),
        (("LRG", 1), ("LSST", 1)),
    ]


def test_sample_combinations_supports_three_collections() -> None:
    """Tests that combinations work beyond simple lens-source pairs."""
    z = np.linspace(0.0, 1.0, 3)

    lenses = {"LRG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)})}
    sources = {"LSST": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)})}
    probes = {
        "shear": make_tomo(
            z=z,
            nz=np.ones(3),
            bins={0: np.ones(3), 1: np.ones(3)},
        )
    }

    combos = sample_combinations(lenses, sources, probes)

    assert combos == [
        (("LRG", 0), ("LSST", 0), ("shear", 0)),
        (("LRG", 0), ("LSST", 0), ("shear", 1)),
    ]


def test_sample_combinations_rejects_no_collections() -> None:
    """Tests that at least one collection is required."""
    with pytest.raises(ValueError, match="At least one sample collection"):
        sample_combinations()


def test_sample_helpers_reject_empty_collection() -> None:
    """Tests that flattened sample helpers require at least one sample."""
    with pytest.raises(ValueError, match="At least one sample"):
        sample_bin_labels({})

    with pytest.raises(ValueError, match="At least one sample"):
        sample_bins({})


def test_sample_helpers_accept_different_z_grids() -> None:
    """Tests that labels and flattened curves do not require matching grids."""
    samples = {
        "BGS": make_tomo(
            z=np.array([0.0, 0.5, 1.0]),
            nz=np.ones(3),
            bins={0: np.ones(3)},
        ),
        "LRG": make_tomo(
            z=np.array([0.0, 0.6, 1.0]),
            nz=np.ones(3),
            bins={0: np.ones(3)},
        ),
    }

    assert sample_bin_labels(samples) == [("BGS", 0), ("LRG", 0)]
    assert list(sample_bins(samples)) == [("BGS", 0), ("LRG", 0)]


def test_combine_parent_nz_interpolation_zeros_outside_input_range() -> None:
    """Tests that interpolation does not extrapolate parent samples."""
    z_target = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    sample = {"z": np.array([0.0, 1.0]), "nz": np.array([2.0, 4.0])}

    _, nz_out = combine_parent_nz(
        [sample],
        interpolate=True,
        z_target=z_target,
    )

    np.testing.assert_allclose(nz_out, np.array([0.0, 2.0, 3.0, 4.0, 0.0]))


def test_combine_parent_nz_uses_target_grid_even_without_interpolation() -> None:
    """Tests that z_target is accepted when it matches the sample grid."""
    z = np.array([0.0, 0.5, 1.0])
    sample = {"z": z, "nz": np.array([1.0, 2.0, 3.0])}

    z_out, nz_out = combine_parent_nz([sample], z_target=z.copy())

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(nz_out, sample["nz"])


def test_combine_parent_nz_rejects_target_grid_without_interpolation_if_different() -> None:
    """Tests that z_target requires interpolation when grids differ."""
    sample = {
        "z": np.array([0.0, 1.0]),
        "nz": np.array([1.0, 2.0]),
    }

    with pytest.raises(ValueError, match="Use interpolate=True"):
        combine_parent_nz(
            [sample],
            z_target=np.array([0.0, 0.5, 1.0]),
        )


def test_interpolate_tomography_zeros_outside_input_range() -> None:
    """Tests that tomography interpolation does not extrapolate."""
    sample = make_tomo(
        z=np.array([0.0, 1.0]),
        nz=np.array([2.0, 4.0]),
        bins={0: np.array([1.0, 3.0])},
    )

    out = interpolate_tomography(
        sample,
        np.array([-1.0, 0.0, 0.5, 1.0, 2.0]),
    )

    np.testing.assert_allclose(out.nz, np.array([0.0, 2.0, 3.0, 4.0, 0.0]))
    np.testing.assert_allclose(out.bins[0], np.array([0.0, 1.0, 2.0, 3.0, 0.0]))


def test_combine_tomographies_preserves_bin_key_order() -> None:
    """Tests that combined tomography bins preserve the first sample key order."""
    z = np.array([0.0, 0.5, 1.0])

    sample_a = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={2: np.ones(3), 0: 2.0 * np.ones(3)},
    )
    sample_b = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={2: 3.0 * np.ones(3), 0: 4.0 * np.ones(3)},
    )

    combined = combine_tomographies([sample_a, sample_b])

    assert list(combined.bins) == [2, 0]
    np.testing.assert_allclose(combined.bins[2], 4.0 * np.ones(3))
    np.testing.assert_allclose(combined.bins[0], 6.0 * np.ones(3))


def test_combine_tomographies_rejects_same_keys_in_different_order() -> None:
    """Tests that bin key order must match exactly."""
    z = np.array([0.0, 0.5, 1.0])

    sample_a = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={0: np.ones(3), 1: np.ones(3)},
    )
    sample_b = make_tomo(
        z=z,
        nz=np.ones(3),
        bins={1: np.ones(3), 0: np.ones(3)},
    )

    with pytest.raises(ValueError, match="same bin keys"):
        combine_tomographies([sample_a, sample_b])


def test_combine_tomographies_rejects_target_grid_without_interpolation_if_different() -> None:
    """Tests that z_target requires interpolation when tomography grids differ."""
    sample = make_tomo(
        z=np.array([0.0, 1.0]),
        nz=np.ones(2),
        bins={0: np.ones(2)},
    )

    with pytest.raises(ValueError, match="Use interpolate=True"):
        combine_tomographies(
            [sample],
            z_target=np.array([0.0, 0.5, 1.0]),
        )


def test_sample_bin_labels_preserve_sample_and_bin_insertion_order() -> None:
    """Tests that flattened labels follow mapping and bin insertion order."""
    z = np.array([0.0, 0.5, 1.0])

    samples = {
        "sample_a": make_tomo(z=z, nz=np.ones(3), bins={2: np.ones(3), 0: np.ones(3)}),
        "sample_b": make_tomo(z=z, nz=np.ones(3), bins={1: np.ones(3)}),
    }

    assert sample_bin_labels(samples) == [
        ("sample_a", 2),
        ("sample_a", 0),
        ("sample_b", 1),
    ]


def test_sample_bins_returns_float_arrays_without_mutating_inputs() -> None:
    """Tests that flattened bins are float arrays copied from input values."""
    z = np.array([0.0, 0.5, 1.0])
    original = np.array([0, 1, 2], dtype=int)

    samples = {
        "sample": make_tomo(
            z=z,
            nz=np.ones(3),
            bins={0: original},
        )
    }

    bins = sample_bins(samples)

    assert bins[("sample", 0)].dtype == float
    np.testing.assert_allclose(bins[("sample", 0)], original)


def test_sample_combinations_with_one_collection_returns_single_labels() -> None:
    """Tests that combinations work for one sample collection."""
    z = np.array([0.0, 0.5, 1.0])
    samples = {
        "sample": make_tomo(
            z=z,
            nz=np.ones(3),
            bins={0: np.ones(3), 1: np.ones(3)},
        )
    }

    combos = sample_combinations(samples)

    assert combos == [
        (("sample", 0),),
        (("sample", 1),),
    ]


def test_sample_combinations_rejects_empty_collection_inside_product() -> None:
    """Tests that empty collections are rejected inside combinations."""
    z = np.array([0.0, 0.5, 1.0])
    samples = {
        "sample": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
    }

    with pytest.raises(ValueError, match="At least one sample"):
        sample_combinations(samples, {})


def test_combine_parent_nz_sums_more_than_two_samples() -> None:
    """Tests that parent n(z) curves sum across more than two samples."""
    z = np.array([0.0, 0.5, 1.0])

    samples = [
        {"z": z, "nz": np.array([1.0, 2.0, 3.0])},
        {"z": z, "nz": np.array([0.5, 1.0, 1.5])},
        {"z": z, "nz": np.array([2.0, 0.0, 1.0])},
    ]

    z_out, nz_out = combine_parent_nz(samples)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(nz_out, np.array([3.5, 3.0, 5.5]))


def test_combine_tomographies_sums_more_than_two_samples() -> None:
    """Tests that tomography bins sum across more than two samples."""
    z = np.array([0.0, 0.5, 1.0])

    sample_a = make_tomo(
        z=z,
        nz=np.array([1.0, 2.0, 3.0]),
        bins={
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
        },
    )
    sample_b = make_tomo(
        z=z,
        nz=np.array([0.5, 1.0, 1.5]),
        bins={
            0: np.array([0.5, 0.5, 0.0]),
            1: np.array([0.0, 0.5, 0.5]),
        },
    )
    sample_c = make_tomo(
        z=z,
        nz=np.array([2.0, 0.0, 1.0]),
        bins={
            0: np.array([2.0, 0.0, 1.0]),
            1: np.array([1.0, 0.0, 0.0]),
        },
    )

    combined = combine_tomographies([sample_a, sample_b, sample_c])

    np.testing.assert_allclose(combined.z, z)
    np.testing.assert_allclose(combined.nz, np.array([3.5, 3.0, 5.5]))
    np.testing.assert_allclose(combined.bins[0], np.array([3.5, 0.5, 1.0]))
    np.testing.assert_allclose(combined.bins[1], np.array([1.0, 1.5, 0.5]))


def test_sample_bin_labels_preserve_three_sample_identities() -> None:
    """Tests that labels preserve identities for more than two samples."""
    z = np.array([0.0, 0.5, 1.0])

    samples = {
        "BGS": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
        "LRG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3), 1: np.ones(3)}),
        "ELG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
    }

    assert sample_bin_labels(samples) == [
        ("BGS", 0),
        ("LRG", 0),
        ("LRG", 1),
        ("ELG", 0),
    ]


def test_sample_combinations_builds_pairs_from_more_than_two_samples_per_collection() -> None:
    """Tests that combinations expand all samples in each collection."""
    z = np.array([0.0, 0.5, 1.0])

    lenses = {
        "BGS": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
        "LRG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
        "ELG": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3)}),
    }
    sources = {
        "LSST": make_tomo(z=z, nz=np.ones(3), bins={0: np.ones(3), 1: np.ones(3)}),
    }

    pairs = sample_combinations(lenses, sources)

    assert pairs == [
        (("BGS", 0), ("LSST", 0)),
        (("BGS", 0), ("LSST", 1)),
        (("LRG", 0), ("LSST", 0)),
        (("LRG", 0), ("LSST", 1)),
        (("ELG", 0), ("LSST", 0)),
        (("ELG", 0), ("LSST", 1)),
    ]


def test_align_bin_grids_preserves_identical_grids() -> None:
    """Tests that matching grids preserve curves and bin labels."""
    z = np.array([0.0, 0.5, 1.0])
    a = {2: np.array([0.0, 1.0, 0.0])}
    b = {5: np.array([1.0, 0.0, 1.0])}

    z_out, a_out, b_out = align_bin_grids(z, a, z.copy(), b)

    np.testing.assert_array_equal(z_out, z)
    np.testing.assert_array_equal(a_out[2], a[2])
    np.testing.assert_array_equal(b_out[5], b[5])
    assert list(a_out) == [2]
    assert list(b_out) == [5]


def test_align_bin_grids_uses_union_grid() -> None:
    """Tests interpolation onto the union of distinct redshift grids."""
    z_a = np.array([0.0, 0.5, 1.0])
    z_b = np.array([0.0, 1.0, 1.5])
    bins_a = {0: np.array([0.0, 2.0, 0.0])}
    bins_b = {3: np.array([0.0, 2.0, 4.0])}

    z, a, b = align_bin_grids(z_a, bins_a, z_b, bins_b)

    np.testing.assert_allclose(z, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(a[0], [0.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(b[3], [0.0, 1.0, 2.0, 4.0])


def test_align_bin_grids_zeros_outside_support() -> None:
    """Tests that union-grid interpolation does not extrapolate."""
    z_a = np.array([0.0, 1.0])
    z_b = np.array([1.0, 2.0])

    z, a, b = align_bin_grids(
        z_a,
        {0: np.array([2.0, 4.0])},
        z_b,
        {1: np.array([6.0, 8.0])},
    )

    np.testing.assert_allclose(z, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(a[0], [2.0, 4.0, 0.0])
    np.testing.assert_allclose(b[1], [0.0, 6.0, 8.0])


def test_sample_combinations_accept_different_z_grids() -> None:
    """Tests that Cartesian bin labels are independent of redshift grids."""
    lenses = {
        "lens": make_tomo(
            z=np.array([0.0, 1.0]),
            nz=np.ones(2),
            bins={0: np.ones(2)},
        )
    }
    sources = {
        "source": make_tomo(
            z=np.array([0.0, 0.5, 1.0]),
            nz=np.ones(3),
            bins={1: np.ones(3)},
        )
    }

    assert sample_combinations(lenses, sources) == [(("lens", 0), ("source", 1))]
