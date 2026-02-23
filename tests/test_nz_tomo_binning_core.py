"""Unit tests for ``binny.nz_tomo.binning_core`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.binning_core import (
    build_bins_on_edges,
    finalize_tomo_metadata,
    resolve_bin_edges,
    validate_bin_edges,
)


def _toy_z_nz(*, n: int = 501):
    """Tests that helper returns a valid (z, nz) pair for parametrized inputs."""
    z = np.linspace(0.0, 2.0, n)
    nz = z**2 * np.exp(-z)
    return z, nz


def _edges_4bins():
    """Tests that helper returns strictly increasing 4-bin edges spanning [0, 2]."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _raw_bin_tophat(z: np.ndarray):
    """Tests that helper returns a raw-bin callback for top-hat selection."""
    z_arr = np.asarray(z, dtype=float)

    def raw_bin_for_edge(i: int, zmin: float, zmax: float) -> np.ndarray:
        _ = i
        mask = (z_arr >= zmin) & (z_arr < zmax)
        return mask.astype(np.float64)

    return raw_bin_for_edge


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
    """Tests that providing bin_edges and binning_scheme raises ValueError."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match=r"either bin_edges or"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=edges,
            binning_scheme="equidistant",
            n_bins=4,
        )


def test_resolve_bin_edges_requires_binning_scheme_when_edges_none():
    """Tests that bin_edges=None requires binning_scheme."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"must provide binning_scheme"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=None,
            n_bins=None,
        )


def test_resolve_bin_edges_string_scheme_requires_n_bins():
    """Tests that string binning_scheme requires n_bins when edges are None."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"must provide n_bins"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme="equidistant",
            n_bins=None,
        )


def test_resolve_bin_edges_equidistant_defaults_to_z_range():
    """Tests that equidistant scheme uses z_axis endpoints by default."""
    z, nz = _toy_z_nz()
    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=4,
    )
    assert edges.shape == (5,)
    assert np.isclose(edges[0], float(z[0]))
    assert np.isclose(edges[-1], float(z[-1]))


def test_resolve_bin_edges_equidistant_aliases_work():
    """Tests that equidistant aliases 'eq' and 'linear' are accepted."""
    z, nz = _toy_z_nz()
    for scheme in ["eq", "linear"]:
        edges = resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=3,
        )
        assert edges.shape == (4,)


def test_resolve_bin_edges_equidistant_bin_range_used():
    """Tests that bin_range overrides z range for equidistant edges."""
    z, nz = _toy_z_nz()
    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=2,
        bin_range=(0.25, 1.25),
    )
    assert np.allclose(edges, np.array([0.25, 0.75, 1.25]))


def test_resolve_bin_edges_equidistant_invalid_bin_range():
    """Tests that invalid bin_range raises a ValueError."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"bin_range must be finite"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme="equidistant",
            n_bins=2,
            bin_range=(1.0, 1.0),
        )


def test_resolve_bin_edges_equal_number_requires_axis_and_weights_both_or_neither():
    """Tests that equal_number requires both equal_number_axis and weights or neither."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"Provide both equal_number_axis"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme="equal_number",
            n_bins=3,
            equal_number_axis=z,
            equal_number_weights=None,
        )


def test_resolve_bin_edges_equal_number_aliases_work():
    """Tests that equal_number aliases 'equipopulated' and 'en' are accepted."""
    z, nz = _toy_z_nz()
    for scheme in ["equipopulated", "en"]:
        edges = resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=4,
        )
        assert edges.shape == (5,)
        assert np.all(np.diff(edges) > 0)


def test_resolve_bin_edges_equal_number_returns_valid_edges():
    """Tests that equal_number returns strictly increasing edges."""
    z, nz = _toy_z_nz()
    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=4,
    )
    assert edges.shape == (5,)
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)


def test_resolve_bin_edges_equal_number_invariant_to_global_weight_scaling():
    """Tests that equal_number edges are invariant to scaling nz when normalization is on."""
    z, nz = _toy_z_nz()
    e1 = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=4,
        normalize_equal_number_weights=True,
    )
    e2 = resolve_bin_edges(
        z_axis=z,
        nz_axis=3.0 * nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=4,
        normalize_equal_number_weights=True,
    )
    assert np.allclose(e1, e2, rtol=0.0, atol=1e-12)


def test_resolve_bin_edges_unsupported_string_scheme():
    """Tests that unsupported string scheme raises a ValueError."""
    z, nz = _toy_z_nz()
    with pytest.raises(ValueError, match=r"Unsupported binning_scheme"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme="nope",
            n_bins=3,
        )


def test_resolve_bin_edges_mixed_mode_rejects_global_n_bins():
    """Tests that mixed binning rejects providing n_bins globally."""
    z, nz = _toy_z_nz()
    segments = [{"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 2.0}]
    with pytest.raises(ValueError, match=r"mixed binning mode"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=segments,
            n_bins=2,
        )


def test_resolve_bin_edges_mixed_dict_requires_segments_key():
    """Tests that mixed binning dict requires key 'segments'."""
    z, nz = _toy_z_nz()
    scheme = {"not_segments": []}
    with pytest.raises(ValueError, match=r"must contain key 'segments'"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=None,
        )


def test_resolve_bin_edges_mixed_requires_sequence_of_segments():
    """Tests that mixed binning rejects non-sequence segments."""
    z, nz = _toy_z_nz()
    scheme = {"segments": "not a list"}
    with pytest.raises(ValueError, match=r"requires a sequence"):
        resolve_bin_edges(
            z_axis=z,
            nz_axis=nz,
            bin_edges=None,
            binning_scheme=scheme,
            n_bins=None,
        )


def test_build_bins_on_edges_basic_no_normalization_no_meta():
    """Tests that build_bins_on_edges builds bins on edges without normalization."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    raw_cb = _raw_bin_tophat(z)

    bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=False,
        norm_method="trapezoid",
        mixer=None,
        need_meta=False,
    )

    assert bins_norms is None
    assert parent_norm is None
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_bins_on_edges_rejects_raw_callback_shape_mismatch():
    """Tests that build_bins_on_edges errors if raw callback returns wrong shape."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def bad_raw(_i: int, _a: float, _b: float) -> np.ndarray:
        return np.zeros(z.size - 1, dtype=float)

    with pytest.raises(ValueError, match=r"returned shape"):
        build_bins_on_edges(
            z=z,
            nz_parent_for_meta=nz,
            bin_edges=edges,
            raw_bin_for_edge=bad_raw,
            normalize_bins=False,
            norm_method="trapezoid",
            mixer=None,
            need_meta=False,
        )


def test_build_bins_on_edges_need_meta_returns_norms_and_parent_norm():
    """Tests that need_meta=True returns bins_norms and parent_norm."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    raw_cb = _raw_bin_tophat(z)

    bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=False,
        norm_method="trapezoid",
        mixer=None,
        need_meta=True,
    )

    assert isinstance(bins_norms, dict)
    assert parent_norm is not None
    assert np.isfinite(parent_norm)
    assert set(bins_norms.keys()) == {0, 1, 2, 3}
    for i in range(4):
        assert np.isfinite(bins_norms[i])


def test_build_bins_on_edges_normalize_bins_yields_unit_integrals_for_nonempty():
    """Tests that normalize_bins=True yields unit integrals for non-empty bins."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def raw_cb(i: int, a: float, b: float) -> np.ndarray:
        _ = i
        mask = (z >= a) & (z < b)
        return (nz * mask.astype(float)).astype(np.float64)

    bins, _, _ = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=True,
        norm_method="trapezoid",
        mixer=None,
        need_meta=False,
    )
    for i in range(4):
        area = float(np.trapezoid(bins[i], x=z))
        assert np.isclose(area, 1.0, rtol=1e-6, atol=1e-10)


def test_build_bins_on_edges_normalize_bins_keeps_zero_bins_unchanged():
    """Tests that normalize_bins=True leaves all-zero bins unchanged."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def raw_cb(i: int, a: float, b: float) -> np.ndarray:
        if i == 2:
            return np.zeros_like(z, dtype=np.float64)
        mask = (z >= a) & (z < b)
        return (nz * mask.astype(float)).astype(np.float64)

    bins, _, _ = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=True,
        norm_method="trapezoid",
        mixer=None,
        need_meta=False,
    )
    assert np.allclose(bins[2], 0.0, atol=0.0, rtol=0.0)


def test_build_bins_on_edges_mixer_is_applied_before_norms_and_normalization():
    """Tests that mixer is applied to raw bins before norms and normalization."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    def raw_cb(i: int, a: float, b: float) -> np.ndarray:
        _ = i
        mask = (z >= a) & (z < b)
        return (nz * mask.astype(float)).astype(np.float64)

    def mixer(raw_bins: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        out = dict(raw_bins)
        out[0] = 2.0 * out[0]
        return out

    bins0, norms0, _ = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=False,
        norm_method="trapezoid",
        mixer=None,
        need_meta=True,
    )
    bins1, norms1, _ = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=False,
        norm_method="trapezoid",
        mixer=mixer,
        need_meta=True,
    )

    assert norms0 is not None and norms1 is not None
    assert norms1[0] == pytest.approx(2.0 * norms0[0])
    assert np.allclose(bins1[0], 2.0 * bins0[0])


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
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=n_bins,
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

    raw_cb = _raw_bin_tophat(z)

    bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z,
        nz_parent_for_meta=nz,
        bin_edges=edges,
        raw_bin_for_edge=raw_cb,
        normalize_bins=True,
        norm_method="trapezoid",
        mixer=None,
        need_meta=True,
    )

    meta = finalize_tomo_metadata(
        kind="photoz",
        z=z,
        parent_nz=nz,
        bin_edges=edges,
        bins=bins,
        inputs={"binning_scheme": "equidistant", "n_bins": n_bins},
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=True,
        save_metadata_path=None,
    )

    frac = meta["bins"]["frac_per_bin"]
    for i in range(n_bins):
        assert frac[i] == pytest.approx(1.0 / n_bins, rel=2e-3)


def test_resolve_bin_edges_mixed_eq_then_equal_number_segment_properties():
    z, nz = _toy_z_nz(n=6001)

    segments = [
        {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
        {"scheme": "equal_number", "n_bins": 2, "z_min": 1.0, "z_max": 2.0},
    ]

    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme=segments,
        n_bins=None,
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
    z, nz = _toy_z_nz(n=6001)

    segments = [
        {"scheme": "equal_number", "n_bins": 3, "z_min": 0.0, "z_max": 1.2},
        {"scheme": "equidistant", "n_bins": 2, "z_min": 1.2, "z_max": 2.0},
    ]

    edges = resolve_bin_edges(
        z_axis=z,
        nz_axis=nz,
        bin_edges=None,
        binning_scheme=segments,
        n_bins=None,
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
