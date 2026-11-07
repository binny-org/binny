"""Benchmark unbinned photometric-redshift fitting across outlier fractions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography
from binny.nz_tomo.photoz import photoz_assignment_probability


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "photoz_unbinned_scan"
)

N_GALAXIES = 100_000
N_REALIZATIONS = 5

OUTLIER_FRACTIONS = np.linspace(0.05, 0.50, 10)

SCATTER_SCALE = 0.03
OUTLIER_SCATTER_SCALE = 0.08
OUTLIER_MEAN_SHIFT = 0.12

N_INIT = 10
MAX_ITER = 1000
TOL = 1e-8

Z = np.linspace(0.0, 3.0, 1000)
BIN_EDGES = np.linspace(0.2, 2.2, 6)

COLUMNS = [
    "outlier_frac",
    "realization",
    "delta_scatter_scale",
    "delta_outlier_scatter_scale",
    "delta_outlier_mean_shift",
    "delta_outlier_frac",
    "max_population_error_percent",
    "max_mean_z_shift",
    "max_l1_distance",
    "converged",
    "n_iter",
    "ks_statistic",
    "loglike",
    "aic",
    "bic",
]


def parent_nz(z: np.ndarray) -> np.ndarray:
    """Return a normalized Smail-like parent redshift distribution."""
    nz = z**2 * np.exp(-((z / 0.8) ** 1.5))
    return nz / np.trapezoid(nz, z)


def generate_catalogue(
    z: np.ndarray,
    nz: np.ndarray,
    outlier_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate catalogue redshifts with injected photometric-redshift errors."""
    rng = np.random.default_rng(seed)

    cdf = np.zeros_like(z)
    cdf[1:] = np.cumsum(
        0.5 * (nz[:-1] + nz[1:]) * np.diff(z)
    )
    cdf /= cdf[-1]

    z_true = np.interp(rng.random(N_GALAXIES), cdf, z)

    is_outlier = rng.random(N_GALAXIES) < outlier_frac

    eps = rng.normal(0.0, SCATTER_SCALE, N_GALAXIES)
    eps[is_outlier] = rng.normal(
        OUTLIER_MEAN_SHIFT,
        OUTLIER_SCATTER_SCALE,
        is_outlier.sum(),
    )

    z_obs = z_true + eps * (1.0 + z_true)

    return z_true, z_obs


def tomography_diagnostics(
    nz: np.ndarray,
    injected: dict[str, float],
    recovered: dict[str, float],
) -> tuple[float, float, float]:
    """Return maximum absolute tomography errors across all bins."""
    population_errors = []
    mean_shifts = []
    l1_distances = []

    for z_min, z_max in zip(
        BIN_EDGES[:-1], BIN_EDGES[1:], strict=True
    ):
        p_true = photoz_assignment_probability(
            Z,
            bin_min=z_min,
            bin_max=z_max,
            **injected,
        )
        p_fit = photoz_assignment_probability(
            Z,
            bin_min=z_min,
            bin_max=z_max,
            **recovered,
        )

        nz_true = nz * p_true
        nz_fit = nz * p_fit

        n_true = np.trapezoid(nz_true, Z)
        n_fit = np.trapezoid(nz_fit, Z)

        population_errors.append(
            100.0 * (n_fit / n_true - 1.0)
        )

        mean_true = np.trapezoid(Z * nz_true, Z) / n_true
        mean_fit = np.trapezoid(Z * nz_fit, Z) / n_fit

        mean_shifts.append(mean_fit - mean_true)

        l1_distances.append(
            np.trapezoid(
                np.abs(nz_fit / n_fit - nz_true / n_true),
                Z,
            )
        )

    return (
        max(abs(x) for x in population_errors),
        max(abs(x) for x in mean_shifts),
        max(l1_distances),
    )


def run_scan() -> np.ndarray:
    """Measure parameter recovery and tomography errors across fractions."""
    nz = parent_nz(Z)
    tomo = NZTomography()

    rows = []

    for outlier_frac in OUTLIER_FRACTIONS:
        injected = tomo.to_binny_photoz_params(
            {
                "scatter_scale": SCATTER_SCALE,
                "outlier_scatter_scale": OUTLIER_SCATTER_SCALE,
                "outlier_mean_shift": OUTLIER_MEAN_SHIFT,
                "outlier_frac": float(outlier_frac),
            }
        )

        for realization in range(N_REALIZATIONS):
            seed = 42 + realization

            z_true, z_obs = generate_catalogue(
                Z,
                nz,
                float(outlier_frac),
                seed,
            )

            fit, metadata = tomo.fit_photoz_errors(
                z_true,
                z_obs,
                n_init=N_INIT,
                max_iter=MAX_ITER,
                tol=TOL,
                seed=seed,
                return_metadata=True,
            )

            recovered = tomo.to_binny_photoz_params(fit)

            pop_error, mean_shift, l1 = tomography_diagnostics(
                nz,
                injected,
                recovered,
            )

            rows.append(
                [
                    outlier_frac,
                    realization,
                    fit["scatter_scale"] - SCATTER_SCALE,
                    fit["outlier_scatter_scale"] - OUTLIER_SCATTER_SCALE,
                    fit["outlier_mean_shift"] - OUTLIER_MEAN_SHIFT,
                    fit["outlier_frac"] - outlier_frac,
                    pop_error,
                    mean_shift,
                    l1,
                    int(metadata["converged"]),
                    metadata["n_iter"],
                    metadata["ks_statistic"],
                    metadata["loglike"],
                    metadata["aic"],
                    metadata["bic"],
                ]
            )

            print(
                f"outlier_frac={outlier_frac:.2f} "
                f"realization={realization} "
                f"delta_outlier_frac={fit['outlier_frac'] - outlier_frac:+.5f} "
                f"max_delta_z={mean_shift:.6f} "
                f"KS={metadata['ks_statistic']:.6f} "
                f"converged={metadata['converged']} "
                f"iterations={metadata['n_iter']}"
            )

    return np.asarray(rows, dtype=float)


def plot_results(results: np.ndarray) -> None:
    """Plot parameter biases, tomography errors, and fit diagnostics."""
    diagnostics = [
        (2, "Core scatter bias", "core_scatter_bias.png"),
        (3, "Tail scatter bias", "tail_scatter_bias.png"),
        (4, "Tail displacement bias", "tail_mean_bias.png"),
        (5, "Outlier fraction bias", "outlier_fraction_bias.png"),
        (
            6,
            "Maximum population error (%)",
            "population_error.png",
        ),
        (
            7,
            "Maximum |mean-redshift shift|",
            "mean_redshift_shift.png",
        ),
        (8, "Maximum L1 distance", "l1_distance.png"),
        (10, "Expectation-maximization iterations", "em_iterations.png"),
        (11, "Kolmogorov-Smirnov statistic", "ks_statistic.png"),
    ]

    for column, ylabel, filename in diagnostics:
        means = []
        stds = []

        for outlier_frac in OUTLIER_FRACTIONS:
            subset = results[np.isclose(results[:, 0], outlier_frac)]
            values = subset[:, column]

            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.errorbar(
            OUTLIER_FRACTIONS,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
        )

        if column in (2, 3, 4, 5):
            ax.axhline(
                0.0,
                color="black",
                lw=1,
                ls=":",
            )

        ax.set(
            xlabel="Injected outlier fraction",
            ylabel=ylabel,
            title=ylabel,
        )

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(fig)


def print_summary(results: np.ndarray) -> None:
    """Print aggregate recovery, convergence, and tomography diagnostics."""
    print("\nUNBINNED PHOTOMETRIC-REDSHIFT FITTING SUMMARY")
    print("-" * 99)
    print(
        f"{'Outlier frac':>12} "
        f"{'Mean delta_f':>14} "
        f"{'Std delta_f':>14} "
        f"{'Max |delta_z|':>15} "
        f"{'Max pop (%)':>13} "
        f"{'Mean KS':>11} "
        f"{'Converged':>10}"
    )
    print("-" * 99)

    for outlier_frac in OUTLIER_FRACTIONS:
        subset = results[np.isclose(results[:, 0], outlier_frac)]

        print(
            f"{outlier_frac:>12.2f} "
            f"{np.mean(subset[:, 5]):>14.6f} "
            f"{np.std(subset[:, 5], ddof=1):>14.6f} "
            f"{np.max(subset[:, 7]):>15.6f} "
            f"{np.max(subset[:, 6]):>13.4f} "
            f"{np.mean(subset[:, 11]):>11.6f} "
            f"{int(np.sum(subset[:, 9])):>7}/{len(subset):<2}"
        )

    print("\nOVERALL DIAGNOSTICS")
    print("-" * 55)
    print(f"Number of fits:          {len(results)}")
    print(
        f"Converged fits:          "
        f"{int(np.sum(results[:, 9]))}/{len(results)}"
    )
    print(
        f"Mean outlier bias:       "
        f"{np.mean(results[:, 5]):+.6f}"
    )
    print(
        f"Maximum |outlier bias|:  "
        f"{np.max(np.abs(results[:, 5])):.6f}"
    )
    print(
        f"Maximum |mean-z shift|:  "
        f"{np.max(results[:, 7]):.6f}"
    )
    print(
        f"Maximum population err:  "
        f"{np.max(results[:, 6]):.4f}%"
    )
    print(
        f"Maximum L1 distance:     "
        f"{np.max(results[:, 8]):.6f}"
    )
    print(
        f"Mean KS statistic:       "
        f"{np.mean(results[:, 11]):.6f}"
    )
    print(
        f"Mean fitting iterations: "
        f"{np.mean(results[:, 10]):.1f}"
    )


def main() -> None:
    """Run the unbinned photometric-redshift outlier-fraction benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_scan()

    np.savetxt(
        OUTPUT_DIR / "results.csv",
        results,
        delimiter=",",
        header=",".join(COLUMNS),
        comments="",
    )

    plot_results(results)
    print_summary(results)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
