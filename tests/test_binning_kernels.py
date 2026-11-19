"""Tests for binning core functionality."""

from __future__ import annotations

import numpy as np
import pytest

from binny.binning.core import build_bins
from binny.binning.edges import resolve_bin_edges, validate_bin_edges
from binny.utils.metadata import finalize_tomo_metadata


def _toy_z_nz(*, n: int = 501):
    """Tests that helper returns a valid (z, nz) pair for parametrized inputs."""
    z = np.linspace(0.0, 2.0, n)
    nz = z**2 * np.exp(-z)
    return z, nz


def _edges_4bins():
    """Tests that helper returns strictly increasing 4-bin edges spanning [0, 2]."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _tophat_assignment(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Returns half-open top-hat selection probabilities."""
    return ((z[None, :] >= edges[:-1, None]) & (z[None, :] < edges[1:, None])).astype(np.float64)


def test_validate_bin_edges_accepts_basic_edges():
    """Tests that validate_bin_edges accepts strictly increasing finite edges."""
    edges = _edges_4bins()
    out = validate_bin_edges(edges)
    assert out.dtype == np.float64
    assert out.ndim == 1
    assert np.allclose(out, edges)


def test_validate_bin_edges_rejects_not_1d():
    """Tests that validate_bin_edges rejects non-1D edges."""
    edges = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    with pytest.raises(ValueError, match=r"bin_edges must be 1D"):
        validate_bin_edges(edges)


def test_validate_bin_edges_rejects_too_short():
    """Tests that validate_bin_edges rejects fewer than two edges."""
    edges = np.array([0.0], dtype=float)
    with pytest.raises(ValueError, match=r"at least two"):
        validate_bin_edges(edges)


def test_validate_bin_edges_rejects_nonfinite():
    """Tests that validate_bin_edges rejects non-finite edges."""
    edges = np.array([0.0, np.nan, 1.0], dtype=float)
    with pytest.raises(ValueError, match=r"finite"):
        validate_bin_edges(edges)


def test_validate_bin_edges_rejects_not_increasing():
    """Tests that validate_bin_edges rejects non-increasing edges."""
    edges = np.array([0.0, 1.0, 0.9, 2.0], dtype=float)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        validate_bin_edges(edges)


def test_validate_bin_edges_require_within_rejects_outside_range():
    """Tests that require_within rejects edges outside [lo, hi]."""
    edges = np.array([-0.1, 0.5, 1.0], dtype=float)
    with pytest.raises(ValueError, match=r"must lie within"):
        validate_bin_edges(edges, require_within=(0.0, 2.0))


def test_validate_bin_edges_require_within_accepts_boundary():
    """Tests that require_within accepts edges exactly on boundaries."""
    edges = np.array([0.0, 0.5, 2.0], dtype=float)
    out = validate_bin_edges(edges, require_within=(0.0, 2.0))
    assert np.allclose(out, edges)


def test_resolve_bin_edges_rejects_edges_and_scheme_together():
    """Tests that explicit edges cannot be combined with a scheme."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={
                "edges": edges,
                "scheme": "equidistant",
                "n_bins": 4,
            },
        )


def test_resolve_bin_edges_requires_binning_specification():
    """Tests that an empty bin specification is rejected."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match="Specify exactly one"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={},
        )


def test_resolve_bin_edges_string_scheme_requires_n_bins():
    """Tests that a generating scheme requires a bin count."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match="n_bins"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": "equidistant"},
        )


def test_resolve_bin_edges_equidistant_defaults_to_z_range():
    """Tests that equidistant edges span the parent redshift grid."""
    z, nz = _toy_z_nz()

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equidistant", "n_bins": 4},
    )

    assert edges.shape == (5,)
    assert np.isclose(edges[0], float(z[0]))
    assert np.isclose(edges[-1], float(z[-1]))


def test_resolve_bin_edges_equidistant_aliases_work():
    """Tests that equidistant aliases are accepted."""
    z, nz = _toy_z_nz()

    for scheme in ["eq", "linear"]:
        edges = resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": scheme, "n_bins": 3},
        )
        assert edges.shape == (4,)


def test_resolve_bin_edges_equidistant_bin_range_used():
    """Tests that the configured range determines equidistant edges."""
    z, nz = _toy_z_nz()

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={
            "scheme": "equidistant",
            "n_bins": 2,
            "range": (0.25, 1.25),
        },
    )

    assert np.allclose(edges, np.array([0.25, 0.75, 1.25]))


def test_resolve_bin_edges_equidistant_invalid_bin_range():
    """Tests rejection of an invalid binning range."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match=r"bins\.range must contain two finite"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={
                "scheme": "equidistant",
                "n_bins": 2,
                "range": (1.0, 1.0),
            },
        )


def test_resolve_bin_edges_equal_number_requires_axis_and_weights_both_or_neither():
    """Tests that an alternative equal-number axis requires its weights."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match="Provide both equal_number_axis"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": "equal_number", "n_bins": 3},
            equal_number_axis=z,
            equal_number_weights=None,
        )


def test_resolve_bin_edges_equal_number_aliases_work():
    """Tests that equal-number aliases are accepted."""
    z, nz = _toy_z_nz()

    for scheme in ["equipopulated", "en"]:
        edges = resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": scheme, "n_bins": 4},
        )
        assert edges.shape == (5,)
        assert np.all(np.diff(edges) > 0)


def test_resolve_bin_edges_equal_number_returns_valid_edges():
    """Tests that equal-number binning returns increasing finite edges."""
    z, nz = _toy_z_nz()

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equal_number", "n_bins": 4},
    )

    assert edges.shape == (5,)
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)


def test_resolve_bin_edges_equal_number_invariant_to_global_weight_scaling():
    """Tests invariance of equal-number edges to global weight scaling."""
    z, nz = _toy_z_nz()

    e1 = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equal_number", "n_bins": 4},
        normalize_equal_number_weights=True,
    )
    e2 = resolve_bin_edges(
        axis=z,
        density=3.0 * nz,
        bins={"scheme": "equal_number", "n_bins": 4},
        normalize_equal_number_weights=True,
    )

    assert np.allclose(e1, e2, rtol=0.0, atol=1e-12)


def test_resolve_bin_edges_unsupported_string_scheme():
    """Tests rejection of unsupported binning schemes."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match="Unsupported binning_scheme"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"scheme": "nope", "n_bins": 3},
        )


def test_resolve_bin_edges_mixed_mode_rejects_global_n_bins():
    """Tests that mixed binning does not accept a global bin count."""
    z, nz = _toy_z_nz()
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 2.0}]

    with pytest.raises(ValueError):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"segments": segments, "n_bins": 2},
        )


def test_resolve_bin_edges_mixed_dict_requires_segments_key():
    """Tests rejection of an invalid mixed-binning specification."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"not_segments": []},
        )


def test_resolve_bin_edges_mixed_requires_sequence_of_segments():
    """Tests that mixed segments must be a sequence."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match=r"bins\.segments must be a sequence"):
        resolve_bin_edges(
            axis=z,
            density=nz,
            bins={"segments": "not a list"},
        )


def test_build_bins_basic_no_normalization_no_meta():
    """Tests that build_bins returns raw distributions without metadata."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins = build_bins(z, nz, edges, _tophat_assignment)

    assert isinstance(bins, dict)
    assert list(bins) == [0, 1, 2, 3]

    selection = _tophat_assignment(z, edges)

    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64
        np.testing.assert_array_equal(bins[i], nz * selection[i])


def test_build_bins_rejects_assignment_shape_mismatch():
    """Tests that build_bins rejects incorrectly shaped selections."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def bad_assignment(axis, boundaries):
        return np.zeros((len(boundaries) - 1, axis.size - 1))

    with pytest.raises(ValueError, match="returned shape"):
        build_bins(z, nz, edges, bad_assignment)


def test_build_bins_returns_raw_integrals():
    """Tests that build_bins returns raw bin and parent integrals."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins, integrals, parent_integral = build_bins(
        z, nz, edges, _tophat_assignment, return_integrals=True
    )

    assert isinstance(integrals, dict)
    assert set(integrals) == {0, 1, 2, 3}
    assert parent_integral == pytest.approx(np.trapezoid(nz, x=z))

    for i in range(4):
        assert integrals[i] == pytest.approx(np.trapezoid(bins[i], x=z))


def test_build_bins_normalizes_nonempty_bins():
    """Tests that normalization produces unit integrals for nonempty bins."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins = build_bins(z, nz, edges, _tophat_assignment, normalize_bins=True)

    for i in range(4):
        assert np.trapezoid(bins[i], x=z) == pytest.approx(1.0)


def test_build_bins_preserves_zero_bins():
    """Tests that normalization leaves zero-mass bins unchanged."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def assignment(axis, boundaries):
        selection = _tophat_assignment(axis, boundaries)
        selection[2] = 0.0
        return selection

    bins, integrals, _ = build_bins(
        z,
        nz,
        edges,
        assignment,
        normalize_bins=True,
        return_integrals=True,
    )

    np.testing.assert_array_equal(bins[2], np.zeros_like(z))
    assert integrals[2] == 0.0


def test_build_bins_weights_applied_before_normalization():
    """Tests that selection weights modify raw distributions and integrals."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def weighted_assignment(axis, boundaries):
        selection = _tophat_assignment(axis, boundaries)
        selection[0] *= 2.0
        return selection

    bins0, norms0, _ = build_bins(z, nz, edges, _tophat_assignment, return_integrals=True)

    bins1, norms1, _ = build_bins(
        z,
        nz,
        edges,
        weighted_assignment,
        selection_kind="weight",
        return_integrals=True,
    )

    normalized, norms2, _ = build_bins(
        z,
        nz,
        edges,
        weighted_assignment,
        selection_kind="weight",
        normalize_bins=True,
        return_integrals=True,
    )

    assert norms1[0] == pytest.approx(2.0 * norms0[0])
    assert norms2 == pytest.approx(norms1)

    np.testing.assert_allclose(bins1[0], 2.0 * bins0[0])
    np.testing.assert_allclose(normalized[0], bins1[0] / norms1[0])

    for i in range(1, 4):
        np.testing.assert_array_equal(bins1[i], bins0[i])
        assert norms1[i] == pytest.approx(norms0[i])


def test_finalize_tomo_metadata_returns_none_when_not_requested():
    """Tests that finalize_tomo_metadata returns None when not requested."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins = {0: nz, 1: nz, 2: nz, 3: nz}

    meta = finalize_tomo_metadata(
        kind="specz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={"x": 1},
        parent_norm=None,
        bins_norms=None,
        include_metadata=False,
        save_metadata_path=None,
    )
    assert meta is None


def test_finalize_tomo_metadata_returns_dict_when_include_metadata_true():
    """Tests that finalize_tomo_metadata returns a dict with expected schema."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins = {0: nz, 1: nz, 2: nz, 3: nz}
    bins_norms = {i: float(np.trapezoid(bins[i], x=z)) for i in range(4)}
    parent_norm = float(np.trapezoid(nz, x=z))

    meta = finalize_tomo_metadata(
        kind="specz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={"normalize_bins": False},
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=True,
        save_metadata_path=None,
    )

    assert isinstance(meta, dict)
    assert meta["kind"] == "specz"
    assert "grid" in meta
    assert "inputs" in meta

    assert "parent_nz" in meta
    assert "norm" in meta["parent_nz"]
    assert meta["parent_nz"]["norm"] == pytest.approx(parent_norm)

    assert "bins" in meta
    b = meta["bins"]
    assert "bin_edges" in b
    assert "bins_norms" in b
    assert "frac_per_bin" in b


def test_finalize_tomo_metadata_frac_per_bin_matches_norms_over_parent():
    """Tests that frac_per_bin equals bins_norms / parent_norm when available."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins = {0: nz, 1: 0.5 * nz, 2: 2.0 * nz, 3: 0.0 * nz}
    bins_norms = {i: float(np.trapezoid(bins[i], x=z)) for i in range(4)}
    parent_norm = float(np.trapezoid(nz, x=z))

    meta = finalize_tomo_metadata(
        kind="photoz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={},
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=True,
        save_metadata_path=None,
    )

    b = meta["bins"]
    assert b["bins_norms"] is not None
    assert b["frac_per_bin"] is not None

    for i in range(4):
        expected = bins_norms[i] / parent_norm if parent_norm != 0.0 else None
        assert b["bins_norms"][i] == pytest.approx(bins_norms[i])
        assert b["frac_per_bin"][i] == pytest.approx(expected)


def test_finalize_tomo_metadata_frac_per_bin_none_when_missing_norms_or_parent():
    """Tests that frac_per_bin is None when parent_norm or bins_norms is missing."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins = {0: nz, 1: nz, 2: nz, 3: nz}

    bins_norms = {i: float(np.trapezoid(bins[i], x=z)) for i in range(4)}
    parent_norm = float(np.trapezoid(nz, x=z))

    meta0 = finalize_tomo_metadata(
        kind="specz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={},
        parent_norm=None,
        bins_norms=bins_norms,
        include_metadata=True,
        save_metadata_path=None,
    )
    assert meta0["parent_nz"]["norm"] is None
    assert meta0["bins"]["bins_norms"] is not None
    assert meta0["bins"]["frac_per_bin"] is None

    meta1 = finalize_tomo_metadata(
        kind="specz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={},
        parent_norm=parent_norm,
        bins_norms=None,
        include_metadata=True,
        save_metadata_path=None,
    )
    assert meta1["parent_nz"]["norm"] == pytest.approx(parent_norm)
    assert meta1["bins"]["bins_norms"] is None
    assert meta1["bins"]["frac_per_bin"] is None


def test_resolve_bin_edges_equal_number_produces_equal_parent_mass_per_bin():
    """Tests that equal_number edges equalize the parent integral per bin."""
    z, nz = _toy_z_nz(n=4001)
    n_bins = 5

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"scheme": "equal_number", "n_bins": n_bins},
    )

    edges = validate_bin_edges(edges, require_within=(float(z[0]), float(z[-1])))

    total = float(np.trapezoid(nz, x=z))
    fracs = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        m = (z >= a) & (z <= b)
        assert m.sum() >= 2
        fracs.append(float(np.trapezoid(nz[m], x=z[m]) / total))

    fracs = np.asarray(fracs, dtype=float)

    assert fracs.sum() == pytest.approx(1.0, rel=2e-3)
    assert np.allclose(fracs, np.full(n_bins, 1.0 / n_bins), atol=2e-3)


def test_finalize_tomo_metadata_uniform_parent_tophat_bins_gives_uniform_frac_per_bin():
    """Tests that uniform parent tophat bins give uniform frac_per_bin."""
    z = np.linspace(0.0, 2.0, 4001)
    nz = np.ones_like(z, dtype=float)  # uniform parent
    n_bins = 4
    edges = np.linspace(0.0, 2.0, n_bins + 1)

    bins, bins_norms, parent_norm = build_bins(
        z,
        nz,
        edges,
        _tophat_assignment,
        normalize_bins=True,
        return_integrals=True,
    )

    meta = finalize_tomo_metadata(
        kind="photoz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={"bins": {"scheme": "equidistant", "n_bins": n_bins}},
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=True,
        save_metadata_path=None,
    )

    frac = meta["bins"]["frac_per_bin"]
    for i in range(n_bins):
        assert frac[i] == pytest.approx(1.0 / n_bins, rel=2e-3)


def test_resolve_bin_edges_mixed_eq_then_equal_number_segment_properties():
    """Tests that equidistant and equal-number segments retain their properties."""
    z, nz = _toy_z_nz(n=6001)

    segments = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
        {"scheme": "equal_number", "n_bins": 2, "z_min": 1.0, "z_max": 2.0},
    ]

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"segments": segments},
    )

    edges = validate_bin_edges(edges, require_within=(0.0, 2.0))

    assert edges.size == 5
    assert edges[0] == pytest.approx(0.0, abs=1e-12)
    assert edges[-1] == pytest.approx(2.0, abs=1e-12)

    # Segment 1 widths
    w0 = edges[1] - edges[0]
    w1 = edges[2] - edges[1]
    assert w0 == pytest.approx(0.5, rel=0.0, abs=1e-10)
    assert w1 == pytest.approx(0.5, rel=0.0, abs=1e-10)

    # Segment 2 equal mass within [1,2]
    def seg_int(a: float, b: float) -> float:
        m = (z >= a) & (z <= b)
        if m.sum() < 2:
            return 0.0
        return float(np.trapezoid(nz[m], x=z[m]))

    Iseg = seg_int(1.0, 2.0)
    Ibin0 = seg_int(edges[2], edges[3])
    Ibin1 = seg_int(edges[3], edges[4])

    assert Ibin0 / Iseg == pytest.approx(0.5, rel=0.0, abs=2e-3)
    assert Ibin1 / Iseg == pytest.approx(0.5, rel=0.0, abs=2e-3)


def test_resolve_bin_edges_mixed_equal_number_then_eq_segment_properties():
    """Tests that reversed mixed segments retain their properties."""
    z, nz = _toy_z_nz(n=6001)

    segments = [
        {"scheme": "equal_number", "n_bins": 3, "z_min": 0.0, "z_max": 1.2},
        {"scheme": "equidistant", "n_bins": 2, "z_min": 1.2, "z_max": 2.0},
    ]

    edges = resolve_bin_edges(
        axis=z,
        density=nz,
        bins={"segments": segments},
    )

    edges = validate_bin_edges(edges, require_within=(0.0, 2.0))

    assert edges.size == 6
    assert edges[0] == pytest.approx(0.0, abs=1e-12)
    assert edges[-1] == pytest.approx(2.0, abs=1e-12)

    # locate join robustly
    j = int(np.argmin(np.abs(edges - 1.2)))
    assert edges[j] == pytest.approx(1.2, rel=0.0, abs=1e-10)

    wA = edges[j + 1] - edges[j]
    wB = edges[j + 2] - edges[j + 1]
    assert wA == pytest.approx(wB, rel=0.0, abs=1e-10)

    # Segment 1 equal mass within [0,1.2]
    def seg_int(a: float, b: float) -> float:
        m = (z >= a) & (z <= b)
        if m.sum() < 2:
            return 0.0
        return float(np.trapezoid(nz[m], x=z[m]))

    Iseg = seg_int(0.0, 1.2)
    ints = [seg_int(edges[i], edges[i + 1]) for i in range(3)]
    for Ii in ints:
        assert Ii / Iseg == pytest.approx(1.0 / 3.0, rel=0.0, abs=2e-3)


@pytest.mark.parametrize(
    ("method", "expected_parent"),
    [("trapezoid", 0.375), ("simpson", 1.0 / 3.0)],
)
def test_build_bins_consistent_quadrature(method, expected_parent):
    """Tests that bin normalization and metadata use the same quadrature."""
    z = np.array([0.0, 0.5, 1.0])
    parent = z**2
    edges = np.array([0.0, 0.5, 1.0])

    bins, integrals, parent_integral = build_bins(
        z,
        parent,
        edges,
        lambda axis, boundaries: np.vstack((np.full_like(axis, 0.25), np.full_like(axis, 0.75))),
        normalize_bins=True,
        norm_method=method,
        return_integrals=True,
    )

    assert parent_integral == pytest.approx(expected_parent)
    assert integrals[0] == pytest.approx(0.25 * expected_parent)
    assert integrals[1] == pytest.approx(0.75 * expected_parent)
    np.testing.assert_allclose(bins[0], parent / expected_parent)
    np.testing.assert_allclose(bins[1], parent / expected_parent)


def test_build_bins_normalizes_small_positive_mass():
    """Tests that small positive bin masses are not treated as empty."""
    z = np.array([0.0, 0.5, 1.0])
    parent = 1e-15 * np.ones_like(z)

    bins, integrals, parent_integral = build_bins(
        z,
        parent,
        np.array([0.0, 1.0]),
        lambda axis, boundaries: np.ones((1, axis.size)),
        normalize_bins=True,
        return_integrals=True,
    )

    np.testing.assert_allclose(bins[0], np.ones_like(z))
    assert integrals[0] == pytest.approx(1e-15, abs=0.0)
    assert parent_integral == pytest.approx(1e-15, abs=0.0)
