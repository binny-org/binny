"""Benchmark photometric-redshift error fitting and tomographic reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography
from binny.nz_tomo.photoz import photoz_assignment_probability


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "photoz_reconstruction"
)

SEED = 42
N_GALAXIES = 100_000

SCATTER_SCALE = 0.03
OUTLIER_SCATTER_SCALE = 0.08
OUTLIER_MEAN_SHIFT = 0.12
OUTLIER_FRAC = 0.20

N_INIT = 10
MAX_ITER = 1000
TOL = 1e-8

Z_MIN = 0.0
Z_MAX = 3.0
N_Z = 1000
BIN_EDGES = np.linspace(0.2, 2.2, 6)


def parent_nz(z: np.ndarray) -> np.ndarray:
    """Return a normalized Smail-like parent redshift distribution."""
    nz = z**2 * np.exp(-((z / 0.8) ** 1.5))
    return nz / np.trapezoid(nz, z)


def generate_catalogue(
    z: np.ndarray,
    nz: np.ndarray,
    *,
    n_galaxies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample true redshifts and inject Gaussian core and tail errors."""
    rng = np.random.default_rng(seed)

    cdf = np.zeros_like(z)
    cdf[1:] = np.cumsum(
        0.5 * (nz[1:] + nz[:-1]) * np.diff(z)
    )
    cdf /= cdf[-1]

    z_true = np.interp(rng.random(n_galaxies), cdf, z)

    is_tail = rng.random(n_galaxies) < OUTLIER_FRAC

    eps = rng.normal(0.0, SCATTER_SCALE, n_galaxies)
    eps[is_tail] = rng.normal(
        OUTLIER_MEAN_SHIFT,
        OUTLIER_SCATTER_SCALE,
        is_tail.sum(),
    )

    z_obs = z_true + eps * (1.0 + z_true)

    return z_true, z_obs


def mixture_pdf(
    eps: np.ndarray,
    *,
    scatter_scale: float,
    outlier_scatter_scale: float,
    outlier_mean_shift: float,
    outlier_frac: float,
) -> np.ndarray:
    """Evaluate the Gaussian core and tail residual probability density.

    Args:
        eps: Normalized photometric-redshift residuals.
        scatter_scale: Standard deviation of the zero-mean Gaussian core.
        outlier_scatter_scale: Standard deviation of the Gaussian tail.
        outlier_mean_shift: Mean displacement of the Gaussian tail.
        outlier_frac: Fraction of objects assigned to the Gaussian tail.

    Returns:
        Probability density evaluated at each residual.
    """
    core = np.exp(-0.5 * (eps / scatter_scale) ** 2) / (
        np.sqrt(2.0 * np.pi) * scatter_scale
    )
    tail = np.exp(
        -0.5 * ((eps - outlier_mean_shift) / outlier_scatter_scale) ** 2
    ) / (
        np.sqrt(2.0 * np.pi) * outlier_scatter_scale
    )

    return (1.0 - outlier_frac) * core + outlier_frac * tail


def construct_tomography(
    z: np.ndarray,
    nz: np.ndarray,
    edges: np.ndarray,
    params: dict[str, float],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Construct normalized tomographic bins and population statistics."""
    bins = []
    populations = []
    means = []

    for z_min, z_max in zip(edges[:-1], edges[1:], strict=True):
        probability = photoz_assignment_probability(
            z,
            bin_min=z_min,
            bin_max=z_max,
            **params,
        )

        nz_bin = nz * probability
        population = np.trapezoid(nz_bin, z)
        mean = np.trapezoid(z * nz_bin, z) / population

        bins.append(nz_bin / population)
        populations.append(population)
        means.append(mean)

    return bins, np.asarray(populations), np.asarray(means)


def plot_parent_distribution(
    z: np.ndarray,
    nz: np.ndarray,
    z_true: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot the input parent distribution and sampled catalogue."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(z, nz, lw=2, label="Input parent n(z)")
    ax.hist(
        z_true,
        bins=80,
        range=(z.min(), z.max()),
        density=True,
        alpha=0.35,
        label="Sampled catalogue",
    )

    ax.set(
        xlabel="True redshift",
        ylabel="Normalized n(z)",
        title="Parent redshift distribution",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "parent_nz.png", dpi=200)
    plt.close(fig)


def plot_residual_fit(
    z_true: np.ndarray,
    z_obs: np.ndarray,
    fit: dict[str, float],
    output_dir: Path,
) -> None:
    """Compare catalogue residuals with injected and fitted mixtures."""
    eps = (z_obs - z_true) / (1.0 + z_true)

    # The histogram is for visualization only, not for fitting.
    x_min = min(-0.3, float(eps.min()))
    x_max = max(0.4, float(eps.max()))
    x = np.linspace(x_min, x_max, 1500)

    injected = mixture_pdf(
        x,
        scatter_scale=SCATTER_SCALE,
        outlier_scatter_scale=OUTLIER_SCATTER_SCALE,
        outlier_mean_shift=OUTLIER_MEAN_SHIFT,
        outlier_frac=OUTLIER_FRAC,
    )
    recovered = mixture_pdf(
        x,
        scatter_scale=fit["scatter_scale"],
        outlier_scatter_scale=fit["outlier_scatter_scale"],
        outlier_mean_shift=fit["outlier_mean_shift"],
        outlier_frac=fit["outlier_frac"],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        eps,
        bins=120,
        range=(x_min, x_max),
        density=True,
        alpha=0.35,
        label="Catalogue residuals",
    )
    ax.plot(x, injected, lw=2, label="Injected model")
    ax.plot(x, recovered, "--", lw=2, label="Fitted model")

    ax.set(
        xlabel=r"$(z_{\rm obs}-z_{\rm true})/(1+z_{\rm true})$",
        ylabel="Probability density",
        title="Photometric-redshift error reconstruction (unbinned fit)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "photoz_residual_fit.png", dpi=200)
    plt.close(fig)


def plot_tomography(
    z: np.ndarray,
    true_bins: list[np.ndarray],
    fitted_bins: list[np.ndarray],
    output_dir: Path,
) -> None:
    """Compare injected and reconstructed tomographic distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(true_bins)))

    for i, (nz_true, nz_fit, color) in enumerate(
        zip(true_bins, fitted_bins, colors, strict=True)
    ):
        ax.plot(
            z,
            nz_true,
            color=color,
            lw=2,
            label=f"Bin {i + 1}",
        )
        ax.plot(
            z,
            nz_fit,
            color=color,
            lw=1.8,
            ls="--",
        )

    ax.plot([], [], color="black", lw=2, label="Injected")
    ax.plot([], [], color="black", ls="--", label="Fitted")

    ax.set(
        xlabel="True redshift",
        ylabel=r"Normalized $n_i(z)$",
        title="Tomographic reconstruction",
        xlim=(Z_MIN, Z_MAX),
    )
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "tomographic_reconstruction.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the photometric-redshift fitting and tomography benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    z = np.linspace(Z_MIN, Z_MAX, N_Z)
    nz = parent_nz(z)

    z_true, z_obs = generate_catalogue(
        z,
        nz,
        n_galaxies=N_GALAXIES,
        seed=SEED,
    )

    tomo = NZTomography()

    fit, metadata = tomo.fit_photoz_errors(
        z_true,
        z_obs,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        tol=TOL,
        seed=SEED,
        return_metadata=True,
    )

    fitted_params = tomo.to_binny_photoz_params(fit)

    injected_params = tomo.to_binny_photoz_params(
        {
            "scatter_scale": SCATTER_SCALE,
            "outlier_scatter_scale": OUTLIER_SCATTER_SCALE,
            "outlier_mean_shift": OUTLIER_MEAN_SHIFT,
            "outlier_frac": OUTLIER_FRAC,
        }
    )

    true_bins, true_pop, true_means = construct_tomography(
        z, nz, BIN_EDGES, injected_params
    )
    fitted_bins, fitted_pop, fitted_means = construct_tomography(
        z, nz, BIN_EDGES, fitted_params
    )

    print("\nPHOTOMETRIC-REDSHIFT PARAMETER RECOVERY")
    print("-" * 85)
    print(
        f"{'Parameter':<24} "
        f"{'Injected':>12} "
        f"{'Fitted':>12} "
        f"{'Difference':>12}"
    )
    print("-" * 85)

    for name, true_value in [
        ("scatter_scale", SCATTER_SCALE),
        ("outlier_scatter_scale", OUTLIER_SCATTER_SCALE),
        ("outlier_mean_shift", OUTLIER_MEAN_SHIFT),
        ("outlier_frac", OUTLIER_FRAC),
    ]:
        print(
            f"{name:<24} "
            f"{true_value:>12.5f} "
            f"{fit[name]:>12.5f} "
            f"{fit[name] - true_value:>+12.5f}"
        )

    print("\nFIT DIAGNOSTICS")
    print("-" * 55)
    print(f"Objects:      {metadata['n']}")
    print(f"Converged:    {metadata['converged']}")
    print(f"Iterations:   {metadata['n_iter']}")
    print(f"Loglike:      {metadata['loglike']:.3f}")
    print(f"AIC:          {metadata['aic']:.3f}")
    print(f"BIC:          {metadata['bic']:.3f}")
    print(f"KS statistic: {metadata['ks_statistic']:.6f}")

    print("\nTOMOGRAPHIC RECONSTRUCTION")
    print("-" * 78)
    print(
        f"{'Bin':<6} {'Population error (%)':>22} "
        f"{'Mean z shift':>18} {'L1 distance':>18}"
    )
    print("-" * 78)

    rows = []

    for i, (nz_true, nz_fit) in enumerate(
        zip(true_bins, fitted_bins, strict=True)
    ):
        pop_error = 100.0 * (
            fitted_pop[i] / true_pop[i] - 1.0
        )
        mean_shift = fitted_means[i] - true_means[i]
        l1 = np.trapezoid(np.abs(nz_fit - nz_true), z)

        rows.append((i + 1, pop_error, mean_shift, l1))

        print(
            f"{i + 1:<6} {pop_error:>22.4f} "
            f"{mean_shift:>18.6f} {l1:>18.6f}"
        )

    plot_parent_distribution(z, nz, z_true, OUTPUT_DIR)
    plot_residual_fit(z_true, z_obs, fit, OUTPUT_DIR)
    plot_tomography(z, true_bins, fitted_bins, OUTPUT_DIR)

    np.savetxt(
        OUTPUT_DIR / "tomography_diagnostics.csv",
        np.asarray(rows),
        delimiter=",",
        header="bin,population_error_percent,mean_z_shift,l1_distance",
        comments="",
    )

    print(f"\nSaved benchmark results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
