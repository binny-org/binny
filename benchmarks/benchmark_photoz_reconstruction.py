
"""Benchmark photo-z error fitting and tomographic reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from binny.api.nz_tomography import NZTomography
from binny.nz_tomo.photoz import photoz_assignment_probability


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42
N_GALAXIES = 100_000

SIGMA_CORE = 0.03
SIGMA_TAIL = 0.08
MEAN_TAIL = 0.12
FRAC_TAIL = 0.20

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
    """Sample true redshifts and inject Gaussian core + tail errors."""
    rng = np.random.default_rng(seed)

    # Sample the parent distribution using a numerical inverse CDF.
    cdf = np.zeros_like(z)
    cdf[1:] = np.cumsum(
        0.5 * (nz[1:] + nz[:-1]) * np.diff(z)
    )
    cdf /= cdf[-1]

    z_true = np.interp(rng.random(n_galaxies), cdf, z)

    # Generate normalized photo-z residuals.
    is_tail = rng.random(n_galaxies) < FRAC_TAIL

    eps = rng.normal(0.0, SIGMA_CORE, n_galaxies)
    eps[is_tail] = rng.normal(
        MEAN_TAIL,
        SIGMA_TAIL,
        is_tail.sum(),
    )

    z_obs = z_true + eps * (1.0 + z_true)

    return z_true, z_obs


def mixture_pdf(
    eps: np.ndarray,
    *,
    sigma_c: float,
    sigma_t: float,
    m_t: float,
    f_t: float,
) -> np.ndarray:
    """Evaluate the Gaussian core + tail residual density."""
    core = np.exp(-0.5 * (eps / sigma_c) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma_c
    )
    tail = np.exp(-0.5 * ((eps - m_t) / sigma_t) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma_t
    )

    return (1.0 - f_t) * core + f_t * tail


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
    """Compare the catalogue residuals with injected and fitted mixtures."""
    eps = (z_obs - z_true) / (1.0 + z_true)
    x = np.linspace(-0.3, 0.3, 1000)

    injected = mixture_pdf(
        x,
        sigma_c=SIGMA_CORE,
        sigma_t=SIGMA_TAIL,
        m_t=MEAN_TAIL,
        f_t=FRAC_TAIL,
    )
    recovered = mixture_pdf(
        x,
        sigma_c=fit["sigma_c"],
        sigma_t=fit["sigma_t"],
        m_t=fit["m_t"],
        f_t=fit["f_t"],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        eps,
        bins=120,
        range=(-0.3, 0.3),
        density=True,
        alpha=0.35,
        label="Catalogue residuals",
    )
    ax.plot(x, injected, lw=2, label="Injected model")
    ax.plot(x, recovered, "--", lw=2, label="Fitted model")

    ax.set(
        xlabel=r"$(z_{\rm obs}-z_{\rm true})/(1+z_{\rm true})$",
        ylabel="Probability density",
        title="Photo-z error reconstruction",
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
    """Compare tomographic redshift distributions."""
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
        xlim=(0.0, 3.0),
    )
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "tomographic_reconstruction.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the photo-z fitting and tomography reconstruction benchmark."""
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

    fit = tomo.fit_photoz_errors(z_true, z_obs)
    fitted_params = tomo.to_binny_photoz_params(fit)

    injected_params = tomo.to_binny_photoz_params(
        {
            "sigma_c": SIGMA_CORE,
            "sigma_t": SIGMA_TAIL,
            "m_t": MEAN_TAIL,
            "f_t": FRAC_TAIL,
        }
    )

    true_bins, true_pop, true_means = construct_tomography(
        z, nz, BIN_EDGES, injected_params
    )
    fitted_bins, fitted_pop, fitted_means = construct_tomography(
        z, nz, BIN_EDGES, fitted_params
    )

    print("\nPHOTO-Z PARAMETER RECOVERY")
    print("-" * 55)
    print(f"{'Parameter':<16} {'Injected':>12} {'Fitted':>12}")
    print("-" * 55)

    for name, true_value in [
        ("sigma_c", SIGMA_CORE),
        ("sigma_t", SIGMA_TAIL),
        ("m_t", MEAN_TAIL),
        ("f_t", FRAC_TAIL),
    ]:
        print(
            f"{name:<16} {true_value:>12.5f} "
            f"{fit[name]:>12.5f}"
        )

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

    # Save numerical diagnostics.
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
