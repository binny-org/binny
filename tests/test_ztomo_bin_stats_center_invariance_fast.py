"""Unit tests for ztomo.bin_stats.bin_centers invariance with fixed edges."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from binny.axes.bin_edges import (
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    geometric_edges,
    log_edges,
)
from binny.ztomo.bin_stats import bin_centers
from binny.ztomo.distributions import (
    gamma_distribution,
    gaussian_distribution,
    gaussian_mixture_distribution,
    lognormal_distribution,
    shifted_smail_distribution,
    skew_normal_distribution,
    smail_like_distribution,
    student_t_distribution,
    tophat_distribution,
)


def _make_z_grid(kind: str, zmin: float, zmax: float, n: int) -> np.ndarray:
    """Generates a redshift grid of specified kind."""
    if kind == "lin":
        return np.linspace(zmin, zmax, n)
    if kind == "log":
        zmin_pos = max(zmin, 1e-4)
        return np.geomspace(zmin_pos, zmax, n)
    raise ValueError(f"Unknown z grid kind: {kind}")


def _n_ref_for(n_bins: int) -> int:
    """Returns number of reference points for a given number of bins."""
    return int(max(2_000, 400 * n_bins))


def _nz_from_distribution(
    z: np.ndarray, name: str, *, sigma_scale: float = 1.0
) -> np.ndarray:
    """Generates n(z) from a named distribution."""
    s = float(sigma_scale)

    if name == "smail":
        return smail_like_distribution(z, z0=0.35, alpha=2.0, beta=1.5)

    if name == "shifted_smail":
        return shifted_smail_distribution(z, z0=0.35, alpha=2.0, beta=1.5, z_shift=0.1)

    if name == "gaussian":
        return gaussian_distribution(z, mu=0.9, sigma=0.25 * s)

    if name == "gaussian_mixture":
        mus = np.array([0.6, 1.4], dtype=float)
        sigmas = np.array([0.18, 0.30], dtype=float) * s
        weights = np.array([0.6, 0.4], dtype=float)
        return gaussian_mixture_distribution(z, mus=mus, sigmas=sigmas, weights=weights)

    if name == "gamma":
        return gamma_distribution(z, k=3.0, theta=0.35 * s)

    if name == "lognormal":
        return lognormal_distribution(z, mu_ln=np.log(0.9), sigma_ln=0.35 * s)

    if name == "tophat":
        return tophat_distribution(z, zmin=0.4, zmax=1.8)

    if name == "skew_normal":
        return skew_normal_distribution(z, xi=0.9, omega=0.35 * s, alpha=4.0)

    if name == "student_t":
        return student_t_distribution(z, mu=0.9, sigma=0.30 * s, nu=5.0)

    raise ValueError(f"Unknown distribution name: {name}")


def _build_bins_from_fixed_edges(
    z: np.ndarray, nz: np.ndarray, edges: np.ndarray
) -> dict[int, np.ndarray]:
    """Builds bins from fixed edges and n(z)."""
    bins: dict[int, np.ndarray] = {}
    for i in range(edges.size - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        if i < edges.size - 2:
            m = (z >= lo) & (z < hi)
        else:
            m = (z >= lo) & (z <= hi)

        bins[i] = nz * m.astype(float)
    return bins


def _bin_integrals(z: np.ndarray, bins: dict[int, np.ndarray]) -> dict[int, float]:
    """Computes integrals of bins over z using trapezoidal rule."""
    return {i: float(np.trapezoid(bins[i], x=z)) for i in bins}


def _common_populated_bins(
    z_ref: np.ndarray,
    bins_ref: dict[int, np.ndarray],
    z: np.ndarray,
    bins: dict[int, np.ndarray],
) -> list[int]:
    """Finds common populated bins between two binnings."""
    ints_ref = _bin_integrals(z_ref, bins_ref)
    ints = _bin_integrals(z, bins)

    vals_ref = np.array(list(ints_ref.values()), dtype=float)
    vals = np.array(list(ints.values()), dtype=float)
    scale = float(
        max(
            np.max(vals_ref) if vals_ref.size else 0.0,
            np.max(vals) if vals.size else 0.0,
        )
    )
    thr = 1e-12 * scale

    pop_ref = {i for i, v in ints_ref.items() if v > thr}
    pop = {i for i, v in ints.items() if v > thr}
    return sorted(pop_ref & pop)


def _chi_of_z(z: np.ndarray) -> np.ndarray:
    """Returns mock comoving distance chi(z)."""
    return z + 0.15 * z**2


def _fixed_edges_from_strategy_on_ref(
    strategy: str,
    z_ref: np.ndarray,
    nz_ref: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Generates fixed bin edges from a strategy applied to reference z and n(z)."""
    zmin = float(z_ref[0])
    zmax = float(z_ref[-1])

    if strategy == "equidistant":
        return equidistant_edges(zmin, zmax, n_bins)

    if strategy == "log":
        return log_edges(max(zmin, 1e-4), zmax, n_bins)

    if strategy == "geometric":
        return geometric_edges(max(zmin, 1e-4), zmax, n_bins)

    if strategy == "equal_number":
        return equal_number_edges(z_ref, nz_ref, n_bins)

    if strategy == "equal_information":
        return equal_information_edges(z_ref, nz_ref, n_bins)

    if strategy == "equidistant_chi":
        chi_ref = _chi_of_z(z_ref)
        return equidistant_chi_edges(z_ref, chi_ref, n_bins)

    raise ValueError(f"Unknown strategy: {strategy}")


def _spacing_tolerance(
    z: np.ndarray, edges: np.ndarray, common_bins: list[int], *, factor: float
) -> float:
    """Computes spacing tolerance based on max dz in common bins."""
    dz = np.diff(z)
    if dz.size == 0:
        return 0.0

    dz_max = 0.0
    for i in common_bins:
        lo = float(edges[i])
        hi = float(edges[i + 1])

        m = (z >= lo) & (z <= hi)
        idx = np.where(m)[0]
        if idx.size < 2:
            continue

        j0, j1 = int(idx[0]), int(idx[-1])
        dz_bin = dz[j0:j1]
        if dz_bin.size:
            dz_max = max(dz_max, float(np.max(dz_bin)))

    return max(factor * dz_max, 5e-3)


@cache
def _z_ref_cached(zmin: float, zmax: float, n_bins: int) -> np.ndarray:
    """Generates cached reference z grid."""
    return np.linspace(zmin, zmax, _n_ref_for(n_bins))


@cache
def _nz_ref_cached(
    distribution: str, sigma_scale: float, zmin: float, zmax: float, n_bins: int
) -> np.ndarray:
    """Generates cached reference n(z)."""
    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    return _nz_from_distribution(z_ref, distribution, sigma_scale=sigma_scale)


@cache
def _edges_cached(
    strategy: str,
    distribution: str,
    sigma_scale: float,
    zmin: float,
    zmax: float,
    n_bins: int,
) -> np.ndarray:
    """Generates cached fixed edges."""
    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    nz_ref = _nz_ref_cached(distribution, sigma_scale, zmin, zmax, n_bins)
    if not np.any(nz_ref > 0):
        return equidistant_edges(zmin, zmax, n_bins)
    return _fixed_edges_from_strategy_on_ref(strategy, z_ref, nz_ref, n_bins)


@cache
def _bins_ref_cached(
    strategy: str,
    distribution: str,
    sigma_scale: float,
    zmin: float,
    zmax: float,
    n_bins: int,
) -> dict[int, np.ndarray]:
    """Generates cached reference bins."""
    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    nz_ref = _nz_ref_cached(distribution, sigma_scale, zmin, zmax, n_bins)
    edges = _edges_cached(strategy, distribution, sigma_scale, zmin, zmax, n_bins)
    return _build_bins_from_fixed_edges(z_ref, nz_ref, edges)


NBINS_FAST = [3, 5, 10]
ZMAX_FAST = [3.0]
N_FAST = [300]

DISTS_FAST = ["gaussian", "smail", "gamma", "tophat"]
STRATS_FAST = ["equidistant", "equal_number", "log", "equidistant_chi"]
CENTERS_FAST = ["mean", "median"]
GRID_FAST = ["lin"]


def _cases_fast_core() -> list[tuple[str, str, str, str, int, float, int]]:
    """Generates fast core test cases."""
    cases: list[tuple[str, str, str, str, int, float, int]] = []
    for dist in DISTS_FAST:
        for cm in CENTERS_FAST:
            for strat in STRATS_FAST:
                for grid in GRID_FAST:
                    for n_bins in NBINS_FAST:
                        for zmax in ZMAX_FAST:
                            for n in N_FAST:
                                zmin = 1e-4 if strat == "log" else 0.0
                                if zmax <= zmin:
                                    continue

                                z_ref = _z_ref_cached(zmin, zmax, n_bins)
                                nz_ref = _nz_ref_cached(dist, 1.0, zmin, zmax, n_bins)
                                if not np.any(nz_ref > 0):
                                    continue

                                edges = _edges_cached(
                                    strat, dist, 1.0, zmin, zmax, n_bins
                                )
                                bins_ref = _bins_ref_cached(
                                    strat, dist, 1.0, zmin, zmax, n_bins
                                )

                                z = _make_z_grid(grid, zmin, zmax, n)
                                nz = _nz_from_distribution(z, dist, sigma_scale=1.0)
                                if not np.any(nz > 0):
                                    continue

                                bins = _build_bins_from_fixed_edges(z, nz, edges)
                                common = _common_populated_bins(
                                    z_ref, bins_ref, z, bins
                                )
                                if not common:
                                    continue

                                cases.append(
                                    (dist, cm, strat, grid, n_bins, float(zmax), int(n))
                                )

    if not cases:
        raise RuntimeError("No fast core cases generated.")
    return cases


def _id_fast(case: tuple[str, str, str, str, int, float, int]) -> str:
    """Generates ID string for fast core test case."""
    dist, cm, strat, grid, n_bins, zmax, n = case
    return f"{dist}-{cm}-{strat}-{grid}-nb{n_bins}-z{zmax:g}-n{n}"


def _cases_fast_smoke() -> list[tuple[str, str, str, int, float, int]]:
    """Generates fast smoke test cases."""
    dists = ["gaussian", "skew_normal", "student_t"]
    strats = ["equidistant", "equal_information"]
    grids = ["lin", "log"]
    cases: list[tuple[str, str, str, int, float, int]] = []
    for dist in dists:
        for strat in strats:
            for grid in grids:
                n_bins = 5
                zmax = 5.0
                n = 400
                zmin = 1e-4 if (grid == "log" or strat == "log") else 0.0
                if zmax <= zmin:
                    continue

                z_ref = _z_ref_cached(zmin, zmax, n_bins)
                nz_ref = _nz_ref_cached(dist, 1.0, zmin, zmax, n_bins)
                if not np.any(nz_ref > 0):
                    continue

                edges = _edges_cached(strat, dist, 1.0, zmin, zmax, n_bins)
                bins_ref = _bins_ref_cached(strat, dist, 1.0, zmin, zmax, n_bins)

                z = _make_z_grid(grid, zmin, zmax, n)
                nz = _nz_from_distribution(z, dist, sigma_scale=1.0)
                if not np.any(nz > 0):
                    continue

                bins = _build_bins_from_fixed_edges(z, nz, edges)
                common = _common_populated_bins(z_ref, bins_ref, z, bins)
                if not common:
                    continue

                cases.append((dist, strat, grid, n_bins, float(zmax), int(n)))

    if not cases:
        raise RuntimeError("No fast smoke cases generated.")
    return cases


def _id_smoke(case: tuple[str, str, str, int, float, int]) -> str:
    """Generates ID string for fast smoke test case."""
    dist, strat, grid, n_bins, zmax, n = case
    return f"{dist}-{strat}-{grid}-nb{n_bins}-z{zmax:g}-n{n}"


_CASES_FAST = _cases_fast_core()
_CASES_SMOKE = _cases_fast_smoke()


@pytest.mark.parametrize(
    "distribution,center_method,strategy,grid_kind,n_bins,zmax,n",
    _CASES_FAST,
    ids=[_id_fast(c) for c in _CASES_FAST],
)
def test_bin_centers_invariant_fixed_edges_fast(
    distribution: str,
    center_method: str,
    strategy: str,
    grid_kind: str,
    n_bins: int,
    zmax: float,
    n: int,
) -> None:
    """Tests that bin centers are invariant under fixed edges."""
    zmin = 1e-4 if strategy == "log" else 0.0

    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    edges = _edges_cached(strategy, distribution, 1.0, zmin, zmax, n_bins)
    bins_ref = _bins_ref_cached(strategy, distribution, 1.0, zmin, zmax, n_bins)

    z = _make_z_grid(grid_kind, zmin, zmax, n)
    nz = _nz_from_distribution(z, distribution, sigma_scale=1.0)
    bins = _build_bins_from_fixed_edges(z, nz, edges)

    common = _common_populated_bins(z_ref, bins_ref, z, bins)

    centers_ref = bin_centers(
        z_ref,
        {i: bins_ref[i] for i in common},
        method=center_method,
        decimal_places=None,
    )
    centers = bin_centers(
        z, {i: bins[i] for i in common}, method=center_method, decimal_places=None
    )

    c_ref = np.array([centers_ref[i] for i in common], dtype=float)
    c = np.array([centers[i] for i in common], dtype=float)

    atol = _spacing_tolerance(z, edges, common, factor=2.0)
    assert np.all(np.isfinite(c))
    assert np.allclose(c, c_ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "distribution,strategy,grid_kind,n_bins,zmax,n",
    _CASES_SMOKE,
    ids=[_id_smoke(c) for c in _CASES_SMOKE],
)
def test_bin_centers_invariant_fixed_edges_smoke(
    distribution: str,
    strategy: str,
    grid_kind: str,
    n_bins: int,
    zmax: float,
    n: int,
) -> None:
    """Tests that bin centers are invariant under fixed edges."""
    center_method = "median"
    zmin = 1e-4 if grid_kind == "log" else 0.0

    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    edges = _edges_cached(strategy, distribution, 1.0, zmin, zmax, n_bins)
    bins_ref = _bins_ref_cached(strategy, distribution, 1.0, zmin, zmax, n_bins)

    z = _make_z_grid(grid_kind, zmin, zmax, n)
    nz = _nz_from_distribution(z, distribution, sigma_scale=1.0)
    bins = _build_bins_from_fixed_edges(z, nz, edges)

    common = _common_populated_bins(z_ref, bins_ref, z, bins)

    centers_ref = bin_centers(
        z_ref,
        {i: bins_ref[i] for i in common},
        method=center_method,
        decimal_places=None,
    )
    centers = bin_centers(
        z, {i: bins[i] for i in common}, method=center_method, decimal_places=None
    )

    c_ref = np.array([centers_ref[i] for i in common], dtype=float)
    c = np.array([centers[i] for i in common], dtype=float)

    atol = _spacing_tolerance(z, edges, common, factor=2.0)
    assert np.allclose(c, c_ref, atol=atol, rtol=0.0)
