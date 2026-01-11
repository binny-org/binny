"""Unit tests for ztomo.bin_stats.bin_centers invariance when edges are fixed.

This runs in slow mode due to the large number of test cases.
"""

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
    schechter_like_distribution,
    shifted_smail_distribution,
    skew_normal_distribution,
    smail_like_distribution,
    student_t_distribution,
    tophat_distribution,
)

pytestmark = pytest.mark.slow

NBINS_LIST = [3, 4, 5, 6, 7, 8, 9, 10]


def _make_z_grid(kind: str, zmin: float, zmax: float, n: int) -> np.ndarray:
    """Generates a z grid of specified kind."""
    if kind == "lin":
        return np.linspace(zmin, zmax, n)
    if kind == "log":
        zmin_pos = max(zmin, 1e-4)
        return np.geomspace(zmin_pos, zmax, n)
    raise ValueError(f"Unknown z grid kind: {kind}")


def _n_ref_for(n_bins: int) -> int:
    """Returns number of reference points for given n_bins."""
    return int(max(4_000, 800 * n_bins))


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

    if name == "schechter":
        return schechter_like_distribution(z, z0=0.6, alpha=1.5)

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
    """Builds bins from fixed edges."""
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
    """Returns integrals of z."""
    return {i: float(np.trapezoid(bins[i], x=z)) for i in bins}


def _common_populated_bins(
    z_ref: np.ndarray,
    bins_ref: dict[int, np.ndarray],
    z: np.ndarray,
    bins: dict[int, np.ndarray],
) -> list[int]:
    """Returns common populated bins."""
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
    """Returns chi(z) for a given z."""
    return z + 0.15 * z**2


def _fixed_edges_from_strategy_on_ref(
    strategy: str,
    z_ref: np.ndarray,
    nz_ref: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Returns fixed edges from strategy on reference."""
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
    z: np.ndarray,
    edges: np.ndarray,
    common_bins: list[int],
    *,
    factor: float,
) -> float:
    """Returns spacing tolerance."""
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
    """Returns reference z grid."""
    return np.linspace(zmin, zmax, _n_ref_for(n_bins))


@cache
def _nz_ref_cached(
    distribution: str,
    sigma_scale: float,
    zmin: float,
    zmax: float,
    n_bins: int,
) -> np.ndarray:
    """Returns reference nz grid."""
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
    """Returns fixed edges from strategy."""
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
    """Returns bins from fixed edges."""
    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    nz_ref = _nz_ref_cached(distribution, sigma_scale, zmin, zmax, n_bins)
    edges = _edges_cached(strategy, distribution, sigma_scale, zmin, zmax, n_bins)
    return _build_bins_from_fixed_edges(z_ref, nz_ref, edges)


_ZMAXS_FULL = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
_ZMAXS_CORE = [1.0, 3.0, 5.0]


def _cases_core() -> list[tuple[str, str, str, str, int, float, int]]:
    """Returns cases for core invariance test."""
    distributions = [
        "smail",
        "shifted_smail",
        "gaussian",
        "gaussian_mixture",
        "gamma",
        "schechter",
        "lognormal",
        "tophat",
        "skew_normal",
        "student_t",
    ]
    center_methods = ["mean", "median", "mode"]
    strategies = [
        "equidistant",
        "log",
        "geometric",
        "equal_number",
        "equal_information",
        "equidistant_chi",
    ]
    grid_kinds = ["lin", "log"]
    n_bins_list = NBINS_LIST
    zmax_list = _ZMAXS_CORE
    n_list = [200, 500, 1500]

    cases: list[tuple[str, str, str, str, int, float, int]] = []

    for distribution in distributions:
        for center_method in center_methods:
            for strategy in strategies:
                for grid_kind in grid_kinds:
                    for n_bins in n_bins_list:
                        for zmax in zmax_list:
                            for n in n_list:
                                if center_method == "mode" and strategy not in {
                                    "equidistant",
                                    "equal_number",
                                }:
                                    continue

                                zmin = (
                                    1e-4
                                    if (
                                        strategy in {"log", "geometric"}
                                        or grid_kind == "log"
                                    )
                                    else 0.0
                                )
                                if zmax <= zmin:
                                    continue

                                z_ref = _z_ref_cached(zmin, zmax, n_bins)
                                nz_ref = _nz_ref_cached(
                                    distribution, 1.0, zmin, zmax, n_bins
                                )
                                if not np.any(nz_ref > 0):
                                    continue

                                edges = _edges_cached(
                                    strategy, distribution, 1.0, zmin, zmax, n_bins
                                )
                                bins_ref = _bins_ref_cached(
                                    strategy, distribution, 1.0, zmin, zmax, n_bins
                                )

                                z = _make_z_grid(grid_kind, zmin, zmax, n)
                                nz = _nz_from_distribution(
                                    z, distribution, sigma_scale=1.0
                                )
                                if not np.any(nz > 0):
                                    continue

                                bins = _build_bins_from_fixed_edges(z, nz, edges)
                                common = _common_populated_bins(
                                    z_ref, bins_ref, z, bins
                                )
                                if not common:
                                    continue

                                cases.append(
                                    (
                                        distribution,
                                        center_method,
                                        strategy,
                                        grid_kind,
                                        n_bins,
                                        float(zmax),
                                        int(n),
                                    )
                                )

    if not cases:
        raise RuntimeError("No cases generated for core invariance test.")
    return cases


def _case_id(case: tuple[str, str, str, str, int, float, int]) -> str:
    """Returns case id for pytest."""
    dist, cm, strat, grid, n_bins, zmax, n = case
    return f"{dist}-{cm}-{strat}-{grid}-nb{n_bins}-z{zmax:g}-n{n}"


def _cases_smoke() -> list[tuple[float, int, str]]:
    """Returns cases for smoke zmax coverage test."""
    cases: list[tuple[float, int, str]] = []

    distribution = "gaussian"
    strategy = "equidistant"
    n = 500

    for zmax in _ZMAXS_FULL:
        for n_bins in [5]:
            for grid_kind in ["lin", "log"]:
                zmin = 1e-4 if grid_kind == "log" else 0.0
                if zmax <= zmin:
                    continue

                z_ref = _z_ref_cached(zmin, zmax, n_bins)
                nz_ref = _nz_ref_cached(distribution, 1.0, zmin, zmax, n_bins)
                if not np.any(nz_ref > 0):
                    continue

                edges = _edges_cached(strategy, distribution, 1.0, zmin, zmax, n_bins)
                bins_ref = _bins_ref_cached(
                    strategy, distribution, 1.0, zmin, zmax, n_bins
                )

                z = _make_z_grid(grid_kind, zmin, zmax, n)
                nz = _nz_from_distribution(z, distribution, sigma_scale=1.0)
                if not np.any(nz > 0):
                    continue

                bins = _build_bins_from_fixed_edges(z, nz, edges)
                common = _common_populated_bins(z_ref, bins_ref, z, bins)
                if not common:
                    continue

                cases.append((float(zmax), int(n_bins), grid_kind))

    if not cases:
        raise RuntimeError("No cases generated for smoke zmax coverage test.")
    return cases


def _cases_param_sweep() -> list[tuple[str, float, str, str, str, int, float, int]]:
    """Returns cases for param sweep test."""
    distributions = ["gaussian", "skew_normal", "student_t", "gamma", "lognormal"]
    sigma_scales = [0.5, 1.0, 2.0]
    center_methods = ["mean", "median"]
    strategies = ["equidistant", "equal_number"]
    grid_kinds = ["lin", "log"]
    n_bins_list = NBINS_LIST
    zmax_list = [3.0, 5.0]
    n_list = [200, 500]

    cases: list[tuple[str, float, str, str, str, int, float, int]] = []

    for distribution in distributions:
        for sigma_scale in sigma_scales:
            for center_method in center_methods:
                for strategy in strategies:
                    for grid_kind in grid_kinds:
                        for n_bins in n_bins_list:
                            for zmax in zmax_list:
                                for n in n_list:
                                    zmin = 1e-4 if grid_kind == "log" else 0.0
                                    if zmax <= zmin:
                                        continue

                                    z_ref = _z_ref_cached(zmin, zmax, n_bins)
                                    nz_ref = _nz_ref_cached(
                                        distribution, sigma_scale, zmin, zmax, n_bins
                                    )
                                    if not np.any(nz_ref > 0):
                                        continue

                                    edges = _edges_cached(
                                        strategy,
                                        distribution,
                                        sigma_scale,
                                        zmin,
                                        zmax,
                                        n_bins,
                                    )
                                    bins_ref = _bins_ref_cached(
                                        strategy,
                                        distribution,
                                        sigma_scale,
                                        zmin,
                                        zmax,
                                        n_bins,
                                    )

                                    z = _make_z_grid(grid_kind, zmin, zmax, n)
                                    nz = _nz_from_distribution(
                                        z, distribution, sigma_scale=sigma_scale
                                    )
                                    if not np.any(nz > 0):
                                        continue

                                    bins = _build_bins_from_fixed_edges(z, nz, edges)
                                    common = _common_populated_bins(
                                        z_ref, bins_ref, z, bins
                                    )
                                    if not common:
                                        continue

                                    cases.append(
                                        (
                                            distribution,
                                            float(sigma_scale),
                                            center_method,
                                            strategy,
                                            grid_kind,
                                            int(n_bins),
                                            float(zmax),
                                            int(n),
                                        )
                                    )

    if not cases:
        raise RuntimeError("No cases generated for param sweep test.")
    return cases


def _case_id_sweep(case: tuple[str, float, str, str, str, int, float, int]) -> str:
    """Returns case id for pytest."""
    dist, s, cm, strat, grid, n_bins, zmax, n = case
    return f"{dist}-s{s:g}-{cm}-{strat}-{grid}-nb{n_bins}-z{zmax:g}-n{n}"


_CASES_CORE = _cases_core()
_CASES_SMOKE = _cases_smoke()
_CASES_SWEEP = _cases_param_sweep()


@pytest.mark.parametrize(
    "distribution,center_method,strategy,grid_kind,n_bins,zmax,n",
    _CASES_CORE,
    ids=[_case_id(c) for c in _CASES_CORE],
)
def test_bin_centers_invariant_when_edges_fixed_all_strategies_and_ranges(
    distribution: str,
    center_method: str,
    strategy: str,
    grid_kind: str,
    n_bins: int,
    zmax: float,
    n: int,
) -> None:
    """Tests that bin centers are invariant under all strategies and ranges."""
    zmin = 1e-4 if (strategy in {"log", "geometric"} or grid_kind == "log") else 0.0

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
        z,
        {i: bins[i] for i in common},
        method=center_method,
        decimal_places=None,
    )

    c_ref = np.array([centers_ref[i] for i in common], dtype=float)
    c = np.array([centers[i] for i in common], dtype=float)

    atol = (
        _spacing_tolerance(z, edges, common, factor=2.0)
        if center_method in {"mean", "median"}
        else _spacing_tolerance(z, edges, common, factor=6.0)
    )

    assert np.all(np.isfinite(c))
    assert np.allclose(c, c_ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "zmax,n_bins,grid_kind",
    _CASES_SMOKE,
    ids=[
        f"z{zmax:g}-nb{n_bins}-{grid_kind}"
        for (zmax, n_bins, grid_kind) in _CASES_SMOKE
    ],
)
def test_bin_centers_invariant_smoke_full_zmax_coverage(
    zmax: float,
    n_bins: int,
    grid_kind: str,
) -> None:
    """Tests that bin centers are invariant for full zmax coverage."""
    distribution = "gaussian"
    strategy = "equidistant"
    center_method = "median"
    n = 500

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
        z,
        {i: bins[i] for i in common},
        method=center_method,
        decimal_places=None,
    )

    c_ref = np.array([centers_ref[i] for i in common], dtype=float)
    c = np.array([centers[i] for i in common], dtype=float)

    atol = _spacing_tolerance(z, edges, common, factor=2.0)
    assert np.allclose(c, c_ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "distribution,sigma_scale,center_method,strategy,grid_kind,n_bins,zmax,n",
    _CASES_SWEEP,
    ids=[_case_id_sweep(c) for c in _CASES_SWEEP],
)
def test_bin_centers_invariant_when_edges_fixed_param_sweep(
    distribution: str,
    sigma_scale: float,
    center_method: str,
    strategy: str,
    grid_kind: str,
    n_bins: int,
    zmax: float,
    n: int,
) -> None:
    """Tests that bin centers are invariant under param sweep."""
    zmin = 1e-4 if grid_kind == "log" else 0.0

    z_ref = _z_ref_cached(zmin, zmax, n_bins)
    edges = _edges_cached(strategy, distribution, sigma_scale, zmin, zmax, n_bins)
    bins_ref = _bins_ref_cached(strategy, distribution, sigma_scale, zmin, zmax, n_bins)

    z = _make_z_grid(grid_kind, zmin, zmax, n)
    nz = _nz_from_distribution(z, distribution, sigma_scale=sigma_scale)
    bins = _build_bins_from_fixed_edges(z, nz, edges)

    common = _common_populated_bins(z_ref, bins_ref, z, bins)

    centers_ref = bin_centers(
        z_ref,
        {i: bins_ref[i] for i in common},
        method=center_method,
        decimal_places=None,
    )
    centers = bin_centers(
        z,
        {i: bins[i] for i in common},
        method=center_method,
        decimal_places=None,
    )

    c_ref = np.array([centers_ref[i] for i in common], dtype=float)
    c = np.array([centers[i] for i in common], dtype=float)

    atol = _spacing_tolerance(z, edges, common, factor=2.0)

    assert np.all(np.isfinite(c))
    assert np.allclose(c, c_ref, atol=atol, rtol=0.0)
