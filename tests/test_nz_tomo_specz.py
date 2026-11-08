"""Unit tests for ``binny.nz_tomo.specz`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.specz import build_specz_bins


def _toy_z_nz(nz_kind: str = "smooth"):
    """Tests that helper returns a valid (z, nz) pair for parametrized inputs."""
    z = np.linspace(0.0, 2.0, 501)
    if nz_kind == "smooth":
        nz = z**2 * np.exp(-z)
    elif nz_kind == "flat":
        nz = np.ones_like(z)
    else:
        raise ValueError("unknown nz_kind")
    return z, nz


def _edges_4bins():
    """Tests that helper returns strictly increasing 4-bin edges spanning [0, 2]."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _assert_col_stochastic(matrix: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-10):
    """Tests that helper asserts a matrix is finite, non-negative, and column-stochastic."""
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= -1e-15)
    col_sums = matrix.sum(axis=0)
    assert np.allclose(col_sums, 1.0, rtol=rtol, atol=atol)


def test_build_specz_bins_keys_shapes_and_dtype():
    """Tests that build_specz_bins returns contiguous integer keys with
    correct shapes."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins = build_specz_bins(z, nz, edges)
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_normalize_bins_true_integrals_are_one():
    """Tests that normalize_bins=True yields unit-normalized non-empty bin shapes."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()
    bins = build_specz_bins(z, nz, edges, normalize_input=True, normalize_bins=True)
    for i in range(4):
        integ = np.trapezoid(bins[i], x=z)
        assert np.isclose(integ, 1.0, rtol=1e-6, atol=1e-10)


def test_build_specz_bins_no_bin_normalization_respects_completeness():
    """Tests that normalize_bins=False preserves relative normalizations
    across bins."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()
    completeness = [1.0, 0.8, 0.6, 0.4]
    bins = build_specz_bins(
        z,
        nz,
        edges,
        completeness=completeness,
        normalize_input=True,
        normalize_bins=False,
    )
    integrals = np.array([np.trapezoid(bins[i], x=z) for i in range(4)])
    assert np.all(integrals >= 0.0)
    assert integrals.sum() <= 1.0 + 1e-8


def test_build_specz_bins_invalid_edges_outside_range():
    """Tests that bin_edges outside the z grid range raises a ValueError."""
    z, nz = _toy_z_nz()
    edges = np.array([-0.1, 0.5, 1.0])
    with pytest.raises(ValueError, match=r"within"):
        build_specz_bins(z, nz, edges)


def test_build_specz_bins_invalid_edges_not_increasing():
    """Tests that non-increasing bin_edges raises a ValueError."""
    z, nz = _toy_z_nz()
    edges = np.array([0.0, 1.0, 0.9, 2.0])
    with pytest.raises(ValueError, match=r"strictly increasing"):
        build_specz_bins(z, nz, edges)


def test_build_specz_bins_catastrophic_neighbor_changes_bins():
    """Tests that enabling catastrophic leakage changes observed-bin
    distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins0 = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )
    bins1 = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.2,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )
    diffs = [np.linalg.norm(bins1[i] - bins0[i]) for i in range(4)]
    assert any(d > 0 for d in diffs)


def test_build_specz_bins_explicit_matrix_overrides_catastrophic_frac():
    """Tests that an explicit response matrix overrides catastrophic_frac and
    leakage_model."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bins = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        response_matrix=matrix,
        normalize_input=True,
        normalize_bins=False,
    )
    mask_01 = (z >= 0.5) & (z < 1.0)
    mask_00 = (z >= 0.0) & (z < 0.5)
    i0 = np.trapezoid(bins[0][mask_01], x=z[mask_01])
    i1 = np.trapezoid(bins[1][mask_00], x=z[mask_00])
    assert i0 > 0.0
    assert i1 > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_const():
    """Tests that enabling constant measurement scatter changes distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins_no = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        normalize_bins=False,
    )
    bins_sc = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        specz_scatter=[5e-4, 5e-4, 8e-4, 1e-3],
        specz_scatter_model="const",
        normalize_bins=False,
    )
    dist = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))
    assert dist > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_sigma0_sigma1():
    """Tests that enabling sigma0_plus_sigma1_1pz measurement scatter changes
    distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins_no = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        normalize_bins=False,
    )
    bins_sc = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        specz_scatter=None,
        specz_scatter_model="sigma0_plus_sigma1_1pz",
        sigma0=1e-4,
        sigma1=2e-4,
        normalize_bins=False,
    )
    dist = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))
    assert dist > 0.0


def test_build_specz_bins_binning_scheme_equidistant_uses_n_bins():
    """Tests that equidistant binning_scheme creates n_bins contiguous outputs."""
    z, nz = _toy_z_nz()
    bins = build_specz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=4,
    )
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_binning_scheme_equidistant_aliases_work():
    """Tests that equidistant binning_scheme aliases are accepted."""
    z, nz = _toy_z_nz()
    for scheme in ["eq", "linear"]:
        bins = build_specz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=3,
        )
        assert list(bins.keys()) == [0, 1, 2]


def test_build_specz_bins_binning_scheme_equal_number_uses_n_bins():
    """Tests that equal_number binning_scheme creates n_bins contiguous outputs."""
    z, nz = _toy_z_nz("smooth")
    bins = build_specz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=4,
        normalize_input=True,
        normalize_bins=False,
    )
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_binning_scheme_equal_number_aliases_work():
    """Tests that equal_number binning_scheme aliases are accepted."""
    z, nz = _toy_z_nz("smooth")
    for scheme in ["equipopulated", "en"]:
        bins = build_specz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=3,
            normalize_input=True,
            normalize_bins=False,
        )
        assert list(bins.keys()) == [0, 1, 2]


def test_build_specz_bins_mixed_segments_sequence_creates_outputs():
    """Tests that mixed segment binning_scheme as a sequence is accepted."""
    z, nz = _toy_z_nz("smooth")
    segments = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
        {"scheme": "equidistant", "n_bins": 2, "z_min": 1.0, "z_max": 2.0},
    ]
    bins = build_specz_bins(z, nz, bin_edges=None, binning_scheme=segments, n_bins=None)
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape


def test_build_specz_bins_mixed_segments_dict_with_segments_key_is_accepted():
    """Tests that mixed segment binning_scheme dict with 'segments' is accepted."""
    z, nz = _toy_z_nz("smooth")
    scheme = {
        "segments": [
            {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
            {"scheme": "equidistant", "n_bins": 1, "z_min": 1.0, "z_max": 2.0},
        ]
    }
    bins = build_specz_bins(z, nz, bin_edges=None, binning_scheme=scheme, n_bins=None)
    assert list(bins.keys()) == [0, 1, 2]
    for i in range(3):
        assert bins[i].shape == z.shape


def test_build_specz_bins_mixed_segments_equal_number_is_accepted():
    """Tests that mixed segments can include equal_number edges."""
    z, nz = _toy_z_nz("smooth")
    segments = [{"scheme": "equal_number", "n_bins": 4, "z_min": 0.0, "z_max": 2.0}]
    bins = build_specz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme=segments,
        n_bins=None,
        normalize_input=True,
        normalize_bins=False,
    )
    assert list(bins.keys()) == [0, 1, 2, 3]


def test_build_specz_bins_rejects_bin_edges_and_binning_scheme_together():
    """Tests that providing bin_edges and binning_scheme raises ValueError."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match=r"either bin_edges or"):
        build_specz_bins(z, nz, bin_edges=edges, binning_scheme="equidistant", n_bins=4)


def test_build_specz_bins_rejects_bin_edges_and_n_bins_together():
    """Tests that providing bin_edges and n_bins raises ValueError."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match=r"either bin_edges or"):
        build_specz_bins(z, nz, bin_edges=edges, n_bins=4)


def test_build_specz_bins_requires_binning_scheme_when_bin_edges_none():
    """Tests that bin_edges=None requires binning_scheme."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"binning_scheme"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme=None)


def test_build_specz_bins_requires_n_bins_for_string_binning_scheme():
    """Tests that string binning_scheme requires n_bins when bin_edges is None."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"n_bins"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme="equidistant", n_bins=None)


def test_build_specz_bins_rejects_unknown_string_binning_scheme():
    """Tests that an unknown string binning_scheme raises ValueError."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"Unsupported binning_scheme"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme="nope", n_bins=4)


def test_build_specz_bins_mixed_mode_rejects_global_n_bins():
    """Tests that mixed binning rejects n_bins when bin_edges is None."""
    z, nz = _toy_z_nz()
    scheme = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 2.0}]
    with pytest.raises(ValueError, match=r"mixed binning"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme=scheme, n_bins=2)


def test_build_specz_bins_mixed_dict_requires_segments_key():
    """Tests that mixed binning dict requires a 'segments' key."""
    z, nz = _toy_z_nz()
    scheme = {"not_segments": []}
    with pytest.raises(ValueError, match=r"must contain key 'segments'"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme=scheme, n_bins=None)


def test_build_specz_bins_mixed_binning_requires_sequence_of_segments():
    """Tests that mixed binning rejects non-sequence segments."""
    z, nz = _toy_z_nz()
    scheme = {"segments": "not a list"}
    with pytest.raises(ValueError, match=r"requires a sequence"):
        build_specz_bins(z, nz, bin_edges=None, binning_scheme=scheme, n_bins=None)


def test_build_specz_bins_writes_metadata_when_save_path_provided(
    tmp_path,
) -> None:
    """Tests that build_specz_bins writes metadata when save_metadata_path is set."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    outpath = tmp_path / "meta.txt"

    bins = build_specz_bins(
        z,
        nz,
        edges,
        include_metadata=False,
        save_metadata_path=str(outpath),
    )
    assert isinstance(bins, dict)
    assert outpath.exists()
    text = outpath.read_text(encoding="utf-8")
    assert "kind" in text or "specz" in text


def test_build_specz_bins_warns_when_zero_catastrophic_frac_and_leakage_params_passed() -> None:
    """Tests that build_specz_bins propagates the zero-catastrophic leakage warning."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    with pytest.warns(
        RuntimeWarning,
        match=r"catastrophic_frac is zero for all bins, so catastrophic leakage parameters",
    ):
        bins = build_specz_bins(
            z,
            nz,
            edges,
            catastrophic_frac=0.0,
            leakage_model="uniform",
            leakage_sigma=3.0,
            normalize_input=True,
            normalize_bins=False,
        )

    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
