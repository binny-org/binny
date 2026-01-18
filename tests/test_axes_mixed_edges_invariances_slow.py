"""Unit tests for binny.axes.mixed_edges invariances.

This test suite focuses on verifying invariance and equivariance properties
of the `mixed_edges` function under various transformations. These tests are
designed to be more computationally intensive and may take longer to run.
They only run in slow test mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from binny.axes.mixed_edges import mixed_edges
from binny.nz_tomo.bin_stats import bin_centers

pytestmark = pytest.mark.slow


def _grid_linear(n: int = 4001, x_min: float = 0.0, x_max: float = 5.0) -> np.ndarray:
    """Returns a linear grid."""
    return np.linspace(float(x_min), float(x_max), int(n))


def _grid_log(n: int = 4001, x_min: float = 1e-4, x_max: float = 5.0) -> np.ndarray:
    """Returns a logarithmic grid."""
    return np.geomspace(float(max(x_min, 1e-8)), float(x_max), int(n))


def _gaussian(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Returns a Gaussian function evaluated at x."""
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _chi_of_z(z: np.ndarray) -> np.ndarray:
    """Returns a toy chi(z) mapping."""
    return z + 0.15 * z**2


def _bins_from_edges_triangular(axis: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    """Returns per-bin arrays with triangular weighting within each edge interval."""
    x = np.asarray(axis, dtype=float)
    e = np.asarray(edges, dtype=float)

    bins: dict[int, np.ndarray] = {}
    for i in range(e.size - 1):
        lo = float(e[i])
        hi = float(e[i + 1])
        mid = 0.5 * (lo + hi)
        width = hi - lo

        t = 1.0 - np.abs((x - mid) / (0.5 * width))
        t = np.clip(t, 0.0, None)

        bins[i] = t

    return bins


def _assert_edges_ok(edges: np.ndarray, n_bins: int, *, x_min: float, x_max: float) -> None:
    """Asserts that edges are valid and match expected endpoints."""
    e = np.asarray(edges, dtype=float)
    assert e.ndim == 1
    assert e.size == n_bins + 1
    assert np.all(np.isfinite(e))
    assert np.all(np.diff(e) > 0)
    assert e[0] == pytest.approx(float(x_min), rel=0.0, abs=1e-12)
    assert e[-1] == pytest.approx(float(x_max), rel=0.0, abs=1e-12)


def _split_interval(
    x_min: float,
    x_max: float,
    n_segments: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Returns random sub-intervals splitting [x_min, x_max]."""
    if n_segments == 1:
        return [(float(x_min), float(x_max))]

    cuts = np.sort(rng.uniform(float(x_min), float(x_max), size=n_segments - 1))
    pts = np.concatenate([[float(x_min)], cuts, [float(x_max)]])
    segs: list[tuple[float, float]] = []
    for i in range(n_segments):
        lo = float(pts[i])
        hi = float(pts[i + 1])

        if hi <= lo:
            hi = lo + 1e-6
        segs.append((lo, hi))

    segs[0] = (float(x_min), segs[0][1])
    segs[-1] = (segs[-1][0], float(x_max))
    return segs


def _choose_nbins_per_segment(total: int, n_segments: int, rng: np.random.Generator) -> list[int]:
    """Returns a random partition of total into n_segments positive integers."""
    if n_segments == 1:
        return [int(total)]
    # sample breakpoints in (0, total)
    cuts = np.sort(rng.choice(np.arange(1, total), size=n_segments - 1, replace=False))
    pts = np.concatenate([[0], cuts, [total]])
    counts = [int(pts[i + 1] - pts[i]) for i in range(n_segments)]
    assert sum(counts) == total
    assert all(c > 0 for c in counts)
    return counts


def _build_interval_segments(
    method: str,
    x_min: float,
    x_max: float,
    total_n_bins: int,
    *,
    n_segments: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Builds interval-based segment specifications."""
    intervals = _split_interval(x_min, x_max, n_segments, rng)
    nbins = _choose_nbins_per_segment(total_n_bins, n_segments, rng)
    segs: list[dict[str, Any]] = []
    for (lo, hi), n in zip(intervals, nbins, strict=True):
        segs.append(
            {
                "method": method,
                "n_bins": int(n),
                "params": {"x_min": lo, "x_max": hi},
            }
        )

    for i in range(1, len(segs)):
        segs[i]["params"]["x_min"] = float(segs[i - 1]["params"]["x_max"])
    return segs


def _build_array_segment(
    method: str,
    n_bins: int,
    *,
    x: np.ndarray,
    w: np.ndarray | None = None,
    info: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Builds an array-driven segment specification."""
    params: dict[str, Any] = {"x": x}
    if method == "equal_number":
        assert w is not None
        params["weights"] = w
    elif method == "equal_information":
        assert info is not None
        params["info_density"] = info
    else:
        raise ValueError(method)
    return [{"method": method, "n_bins": int(n_bins), "params": params}]


def _build_chi_segment(n_bins: int, *, z: np.ndarray, chi: np.ndarray) -> list[dict[str, Any]]:
    """Builds an equidistant_chi segment specification."""
    return [
        {
            "method": "equidistant_chi",
            "n_bins": int(n_bins),
            "params": {"z": z, "chi": chi},
        }
    ]


@dataclass(frozen=True)
class IntervalCase:
    """Dataclass for interval-method test cases."""

    method: str
    x_min: float
    x_max: float
    total_n_bins: int
    n_segments: int
    seed: int


def _interval_cases() -> list[IntervalCase]:
    """Generates a list of interval-method test cases."""
    methods = ["equidistant", "log", "geometric"]
    total_bins_list = [5, 7, 9, 12]
    seg_counts = [1, 2, 3, 4]
    seeds = list(range(20))

    cases: list[IntervalCase] = []
    for method in methods:
        for total in total_bins_list:
            for nseg in seg_counts:
                for seed in seeds:
                    if nseg > total:
                        continue
                    x_min = 0.0
                    x_max = 5.0

                    if method in {"log", "geometric"}:
                        x_min = 1e-3
                    cases.append(IntervalCase(method, x_min, x_max, total, nseg, seed))
    if not cases:
        raise RuntimeError("No interval cases generated.")
    return cases


def _id_interval(c: IntervalCase) -> str:
    """Generates a unique ID string for an IntervalCase."""
    return f"{c.method}-bins{c.total_n_bins}-segs{c.n_segments}-seed{c.seed}"


@pytest.mark.parametrize("case", _interval_cases(), ids=_id_interval)
def test_mixed_edges_interval_methods_equivariance(case: IntervalCase) -> None:
    """Tests that interval-based mixed_edges are equivariant under affine transforms."""
    rng = np.random.default_rng(case.seed)

    segs = _build_interval_segments(
        case.method,
        case.x_min,
        case.x_max,
        case.total_n_bins,
        n_segments=case.n_segments,
        rng=rng,
    )
    e0 = mixed_edges(segs, total_n_bins=case.total_n_bins)

    a = float(rng.uniform(0.2, 5.0))

    if case.method == "equidistant":
        b = float(rng.uniform(-3.0, 3.0))
    else:
        b = 0.0

    segs_t: list[dict[str, Any]] = []
    for seg in segs:
        p = dict(seg["params"])
        p["x_min"] = a * float(p["x_min"]) + b
        p["x_max"] = a * float(p["x_max"]) + b
        segs_t.append({"method": seg["method"], "n_bins": seg["n_bins"], "params": p})

    e1 = mixed_edges(segs_t, total_n_bins=case.total_n_bins)

    assert np.allclose(e1, a * e0 + b, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize("n_bins", [3, 4, 5, 6, 7, 8, 10, 12])
@pytest.mark.parametrize("grid_kind", ["lin", "log"])
@pytest.mark.parametrize("seed", range(30))
def test_mixed_edges_equal_number_invariant_under_weight_scaling_and_normalization(
    n_bins: int,
    grid_kind: str,
    seed: int,
) -> None:
    """Tests that equal_number mixed_edges are invariant under weight scaling
    and normalization."""
    rng = np.random.default_rng(seed)

    x = _grid_linear() if grid_kind == "lin" else _grid_log()
    mu = float(rng.uniform(float(x.min()) + 0.2, float(x.max()) - 0.2))
    sig = float(rng.uniform(0.15, 0.8))
    w = _gaussian(x, mu, sig) + 1e-12  # strictly positive

    seg = _build_array_segment("equal_number", n_bins, x=x, w=w)

    e0 = mixed_edges(seg, total_n_bins=n_bins)

    scale = float(rng.uniform(0.1, 20.0))
    e1 = mixed_edges(
        _build_array_segment("equal_number", n_bins, x=x, w=scale * w),
        total_n_bins=n_bins,
    )

    norm = float(np.trapezoid(w, x=x))
    w_norm = w / norm
    e2 = mixed_edges(
        _build_array_segment("equal_number", n_bins, x=x, w=w_norm),
        total_n_bins=n_bins,
    )

    assert np.allclose(e0, e1, rtol=0.0, atol=5e-12)
    assert np.allclose(e0, e2, rtol=0.0, atol=5e-12)


@pytest.mark.parametrize("n_bins", [3, 4, 5, 6, 7, 8, 10, 12])
@pytest.mark.parametrize("grid_kind", ["lin", "log"])
@pytest.mark.parametrize("seed", range(30))
def test_mixed_edges_equal_info_invariant_under_density_scaling_and_normalization(
    n_bins: int,
    grid_kind: str,
    seed: int,
) -> None:
    """Tests that equal_information mixed_edges are invariant under density
    scaling/normalization."""
    rng = np.random.default_rng(seed)

    x = _grid_linear() if grid_kind == "lin" else _grid_log()
    mu = float(rng.uniform(float(x.min()) + 0.2, float(x.max()) - 0.2))
    sig = float(rng.uniform(0.15, 0.8))
    info = _gaussian(x, mu, sig) + 1e-12

    seg = _build_array_segment("equal_information", n_bins, x=x, info=info)
    e0 = mixed_edges(seg, total_n_bins=n_bins)

    scale = float(rng.uniform(0.1, 20.0))
    e1 = mixed_edges(
        _build_array_segment("equal_information", n_bins, x=x, info=scale * info),
        total_n_bins=n_bins,
    )

    norm = float(np.trapezoid(info, x=x))
    info_norm = info / norm
    e2 = mixed_edges(
        _build_array_segment("equal_information", n_bins, x=x, info=info_norm),
        total_n_bins=n_bins,
    )

    assert np.allclose(e0, e1, rtol=0.0, atol=5e-12)
    assert np.allclose(e0, e2, rtol=0.0, atol=5e-12)


@pytest.mark.parametrize("method", ["equal_number", "equal_information"])
@pytest.mark.parametrize("n_bins", [4, 6, 8, 10])
@pytest.mark.parametrize("seed", range(20))
def test_mixed_edges_array_methods_invariant_under_grid_refinement(
    method: str,
    n_bins: int,
    seed: int,
) -> None:
    """Tests that array-based mixed_edges are invariant under grid refinement."""
    rng = np.random.default_rng(seed)

    x0 = _grid_linear(n=2001, x_min=0.0, x_max=5.0)
    x1 = _grid_linear(n=12001, x_min=0.0, x_max=5.0)

    mu = float(rng.uniform(0.8, 4.2))
    sig = float(rng.uniform(0.2, 0.7))

    base0 = _gaussian(x0, mu, sig) + 1e-12
    base1 = _gaussian(x1, mu, sig) + 1e-12

    if method == "equal_number":
        seg0 = _build_array_segment(method, n_bins, x=x0, w=base0)
        seg1 = _build_array_segment(method, n_bins, x=x1, w=base1)
    else:
        seg0 = _build_array_segment(method, n_bins, x=x0, info=base0)
        seg1 = _build_array_segment(method, n_bins, x=x1, info=base1)

    e0 = mixed_edges(seg0, total_n_bins=n_bins)
    e1 = mixed_edges(seg1, total_n_bins=n_bins)

    dx = float(np.max(np.diff(x0)))
    assert np.allclose(e1, e0, rtol=0.0, atol=5.0 * dx)


@pytest.mark.parametrize("n_bins", [4, 6, 8, 10, 12])
@pytest.mark.parametrize("seed", range(40))
def test_mixed_edges_equidistant_chi_invariant_under_affine_chi_transform(
    n_bins: int,
    seed: int,
) -> None:
    """Tests that equidistant_chi mixed_edges are invariant under transforms of chi."""
    rng = np.random.default_rng(seed)

    z = _grid_linear(n=6001, x_min=0.0, x_max=3.0)
    chi = _chi_of_z(z)

    a = float(rng.uniform(0.2, 10.0))
    b = float(rng.uniform(-50.0, 50.0))

    e0 = mixed_edges(_build_chi_segment(n_bins, z=z, chi=chi), total_n_bins=n_bins)
    e1 = mixed_edges(_build_chi_segment(n_bins, z=z, chi=a * chi + b), total_n_bins=n_bins)

    assert np.allclose(e0, e1, rtol=0.0, atol=5e-12)


def _bins_from_edges_uniform(axis: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    """Builds per-bin arrays with uniform weighting within each edge interval."""
    z = np.asarray(axis, dtype=float)
    e = np.asarray(edges, dtype=float)

    bins: dict[int, np.ndarray] = {}
    for i in range(e.size - 1):
        lo = float(e[i])
        hi = float(e[i + 1])

        if i < e.size - 2:
            m = (z >= lo) & (z < hi)
        else:
            m = (z >= lo) & (z <= hi)

        bins[i] = m.astype(float)
    return bins


@pytest.mark.parametrize("center_method", ["mean", "median"])
@pytest.mark.parametrize("seed", range(25))
def test_bin_centers_from_mixed_edges_affine_invariance_equidistant(
    center_method: str,
    seed: int,
) -> None:
    """Tests that bin centers from equidistant mixed_edges are affine invariant."""
    rng = np.random.default_rng(seed)

    method = "equidistant"
    total_n_bins = int(rng.choice([5, 7, 9, 12]))
    n_segments = int(rng.choice([1, 2, 3, 4]))
    n_segments = min(n_segments, total_n_bins)

    x_min = 0.0
    x_max = 5.0

    segs = _build_interval_segments(
        method,
        x_min,
        x_max,
        total_n_bins,
        n_segments=n_segments,
        rng=rng,
    )

    edges0 = mixed_edges(segs, total_n_bins=total_n_bins)
    axis0 = _grid_linear(n=8001, x_min=x_min, x_max=x_max)
    bins0 = _bins_from_edges_uniform(axis0, edges0)
    c0d = bin_centers(axis0, bins0, method=center_method, decimal_places=None)
    c0 = np.array([c0d[i] for i in range(total_n_bins)], dtype=float)

    a = float(rng.uniform(0.2, 5.0))
    b = float(rng.uniform(-3.0, 3.0))

    segs_ab: list[dict[str, Any]] = []
    for seg in segs:
        p = dict(seg["params"])
        p["x_min"] = a * float(p["x_min"]) + b
        p["x_max"] = a * float(p["x_max"]) + b
        segs_ab.append({"method": seg["method"], "n_bins": seg["n_bins"], "params": p})

    edges1 = mixed_edges(segs_ab, total_n_bins=total_n_bins)
    axis1 = a * axis0 + b
    bins1 = _bins_from_edges_uniform(axis1, edges1)
    c1d = bin_centers(axis1, bins1, method=center_method, decimal_places=None)
    c1 = np.array([c1d[i] for i in range(total_n_bins)], dtype=float)

    assert np.allclose(c1, a * c0 + b, rtol=0.0, atol=5e-12)


@pytest.mark.parametrize("center_method", ["mean", "median"])
@pytest.mark.parametrize("method", ["log", "geometric"])
@pytest.mark.parametrize("seed", range(25))
def test_bin_centers_from_mixed_edges_scale_invariance_log_geometric(
    center_method: str,
    method: str,
    seed: int,
) -> None:
    """Tests that bin centers from log/geometric mixed_edges are scale invariant."""
    rng = np.random.default_rng(seed)

    total_n_bins = int(rng.choice([5, 7, 9, 12]))
    n_segments = int(rng.choice([1, 2, 3, 4]))
    n_segments = min(n_segments, total_n_bins)

    x_min = 1e-3
    x_max = 5.0

    segs = _build_interval_segments(
        method,
        x_min,
        x_max,
        total_n_bins,
        n_segments=n_segments,
        rng=rng,
    )

    edges0 = mixed_edges(segs, total_n_bins=total_n_bins)

    axis0 = _grid_linear(n=20001, x_min=x_min, x_max=x_max)

    bins0 = _bins_from_edges_triangular(axis0, edges0)
    c0d = bin_centers(axis0, bins0, method=center_method, decimal_places=None)
    c0 = np.array([c0d[i] for i in range(total_n_bins)], dtype=float)

    a = float(rng.uniform(0.2, 5.0))

    segs_a: list[dict[str, Any]] = []
    for seg in segs:
        p = dict(seg["params"])
        p["x_min"] = a * float(p["x_min"])
        p["x_max"] = a * float(p["x_max"])
        segs_a.append({"method": seg["method"], "n_bins": seg["n_bins"], "params": p})

    edges1 = mixed_edges(segs_a, total_n_bins=total_n_bins)
    axis1 = a * axis0

    bins1 = _bins_from_edges_triangular(axis1, edges1)
    c1d = bin_centers(axis1, bins1, method=center_method, decimal_places=None)
    c1 = np.array([c1d[i] for i in range(total_n_bins)], dtype=float)

    dx0 = float(np.max(np.diff(axis0)))
    atol = max(50.0 * dx0, 1e-10)

    assert np.allclose(c1, a * c0, rtol=0.0, atol=atol)


@pytest.mark.parametrize("center_method", ["mean", "median"])
@pytest.mark.parametrize("seed", range(30))
def test_bin_centers_from_mixed_edges_weight_invariance_equal_number(
    center_method: str,
    seed: int,
) -> None:
    """Tests that for equal_number edges, weight scaling does not change edges,
    so centers computed from uniform-per-bin density must be identical."""
    rng = np.random.default_rng(seed)

    n_bins = int(rng.choice([4, 6, 8, 10, 12]))
    x = _grid_linear(n=12001, x_min=0.0, x_max=5.0)

    mu = float(rng.uniform(0.8, 4.2))
    sig = float(rng.uniform(0.2, 0.7))
    w = _gaussian(x, mu, sig) + 1e-12

    edges0 = mixed_edges(
        [{"method": "equal_number", "n_bins": n_bins, "params": {}}],
        x=x,
        weights=w,
        total_n_bins=n_bins,
    )
    edges1 = mixed_edges(
        [{"method": "equal_number", "n_bins": n_bins, "params": {}}],
        x=x,
        weights=7.3 * w,
        total_n_bins=n_bins,
    )

    bins0 = _bins_from_edges_uniform(x, edges0)
    bins1 = _bins_from_edges_uniform(x, edges1)

    c0 = bin_centers(x, bins0, method=center_method, decimal_places=None)
    c1 = bin_centers(x, bins1, method=center_method, decimal_places=None)

    v0 = np.array([c0[i] for i in range(n_bins)], dtype=float)
    v1 = np.array([c1[i] for i in range(n_bins)], dtype=float)

    assert np.allclose(edges0, edges1, rtol=0.0, atol=5e-12)
    assert np.allclose(v0, v1, rtol=0.0, atol=5e-12)


@pytest.mark.parametrize("center_method", ["mean", "median"])
@pytest.mark.parametrize("seed", range(30))
def test_bin_centers_from_mixed_edges_density_invariance_equal_information(
    center_method: str,
    seed: int,
) -> None:
    """Tests that for equal_information edges, density scaling does not change edges,
    so centers computed from uniform-per-bin density must be identical."""
    rng = np.random.default_rng(seed)

    n_bins = int(rng.choice([4, 6, 8, 10, 12]))
    x = _grid_linear(n=12001, x_min=0.0, x_max=5.0)

    mu = float(rng.uniform(0.8, 4.2))
    sig = float(rng.uniform(0.2, 0.7))
    info = _gaussian(x, mu, sig) + 1e-12

    edges0 = mixed_edges(
        [{"method": "equal_information", "n_bins": n_bins, "params": {}}],
        x=x,
        info_density=info,
        total_n_bins=n_bins,
    )
    edges1 = mixed_edges(
        [{"method": "equal_information", "n_bins": n_bins, "params": {}}],
        x=x,
        info_density=11.0 * info,
        total_n_bins=n_bins,
    )

    bins0 = _bins_from_edges_uniform(x, edges0)
    bins1 = _bins_from_edges_uniform(x, edges1)

    c0 = bin_centers(x, bins0, method=center_method, decimal_places=None)
    c1 = bin_centers(x, bins1, method=center_method, decimal_places=None)

    v0 = np.array([c0[i] for i in range(n_bins)], dtype=float)
    v1 = np.array([c1[i] for i in range(n_bins)], dtype=float)

    assert np.allclose(edges0, edges1, rtol=0.0, atol=5e-12)
    assert np.allclose(v0, v1, rtol=0.0, atol=5e-12)
