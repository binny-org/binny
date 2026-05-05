"""Fit, interpolate, and extrapolate fixed-alpha Smail parameters for Roman n(z)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


WENZL22_SCENARIOS: dict[str, dict[str, Any]] = {
    "wenzl22_hls_optimistic": {
        "survey_area_deg2": 2000.0,
        "source_neff": 51.0,
        "lens_neff": 66.0,
        "photoz_scatter": 0.01,
        "n_tomo_bins": 10,
        "binning_scheme": "equal_number",
    },
    "wenzl22_hls_conservative": {
        "survey_area_deg2": 2000.0,
        "source_neff": 51.0,
        "lens_neff": 66.0,
        "photoz_scatter": 0.05,
        "n_tomo_bins": 10,
        "binning_scheme": "equal_number",
    },
    "wenzl22_wide": {
        "survey_area_deg2": 18000.0,
        "source_neff": 43.0,
        "lens_neff": 59.0,
        "photoz_scatter": 0.02,
        "n_tomo_bins": 10,
        "binning_scheme": "equal_number",
    },
}


def neff_to_label(neff: float | int) -> str:
    """Convert n_eff to a stable filename label."""
    value = float(neff)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def read_nz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column z, n(z) file."""
    table = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["z", "nz"],
        engine="python",
    )

    z = table["z"].to_numpy(dtype=float)
    nz = np.clip(table["nz"].to_numpy(dtype=float), 0.0, None)

    if np.sum(nz) <= 0.0:
        raise ValueError(f"n(z) is zero everywhere in {path}")

    return z, nz


def fit_smail_fixed_alpha_from_nz(
    z: np.ndarray,
    nz: np.ndarray,
    *,
    alpha: float,
    z_max: float,
    min_n: int,
) -> dict[str, Any]:
    """Fit n(z) ∝ z**alpha exp[-(z / z0)**beta] directly to tabulated n(z)."""
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)

    mask = np.isfinite(z) & np.isfinite(nz) & (z >= 0.0) & (z <= z_max)
    z = z[mask]
    nz = np.clip(nz[mask], 0.0, None)

    if z.size < min_n:
        return {
            "ok": False,
            "n": z.size,
            "reason": "too_few_grid_points",
        }

    target_norm = np.trapezoid(nz, z)

    if target_norm <= 0.0 or not np.isfinite(target_norm):
        return {
            "ok": False,
            "n": z.size,
            "reason": "invalid_nz_norm",
        }

    target_pdf = nz / target_norm

    def model_pdf(beta: float, z0: float) -> np.ndarray:
        pdf = z**alpha * np.exp(-((z / z0) ** beta))
        norm = np.trapezoid(pdf, z)

        if norm <= 0.0 or not np.isfinite(norm):
            return np.full_like(z, np.nan)

        return pdf / norm

    def loss(params: np.ndarray) -> float:
        beta, z0 = params

        if beta <= 0.0 or z0 <= 0.0:
            return np.inf

        model = model_pdf(beta, z0)

        if not np.all(np.isfinite(model)):
            return np.inf

        return float(np.trapezoid((model - target_pdf) ** 2, z))

    result = minimize(
        loss,
        x0=np.array([0.8, 0.2]),
        bounds=[(0.01, 5.0), (0.001, z_max)],
        method="L-BFGS-B",
    )

    if not result.success:
        return {
            "ok": False,
            "n": z.size,
            "reason": "fit_failed",
            "message": str(result.message),
        }

    beta, z0 = result.x

    return {
        "ok": True,
        "n": z.size,
        "params": {
            "alpha": float(alpha),
            "beta": float(beta),
            "z0": float(z0),
        },
        "fun": float(result.fun),
    }


def fit_one_neff(
    base_dir: Path,
    neff: float | int,
    *,
    alpha: float,
    z_max: float,
    min_n: int,
) -> dict[str, Any]:
    """Fit one Roman n_eff file if it exists."""
    neff_label = neff_to_label(neff)
    path = base_dir / f"roman_n_eff{neff_label}.nz"

    if not path.exists():
        return {
            "ok": False,
            "neff": float(neff),
            "path": str(path),
            "reason": "file_not_found",
        }

    z, nz = read_nz(path)

    result = fit_smail_fixed_alpha_from_nz(
        z,
        nz,
        alpha=alpha,
        z_max=z_max,
        min_n=min_n,
    )

    if not result["ok"]:
        return {
            "ok": False,
            "neff": float(neff),
            "path": str(path),
            "n": result.get("n"),
            "reason": result.get("reason"),
            "message": result.get("message"),
        }

    params = result["params"]

    return {
        "ok": True,
        "neff": float(neff),
        "n": result["n"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "z0": params["z0"],
        "fun": result["fun"],
        "path": str(path),
        "method": "fit",
    }


def linear_interp_or_extrap(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
    *,
    allow_extrapolate: bool,
) -> tuple[float, str]:
    """Linearly interpolate or extrapolate one parameter."""
    if xp.size < 2:
        raise ValueError("At least two successful anchor fits are needed.")

    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    if xp.min() <= x <= xp.max():
        return float(np.interp(x, xp, fp)), "interpolated"

    if not allow_extrapolate:
        raise ValueError(
            f"n_eff={x} is outside the fitted range [{xp.min()}, {xp.max()}]."
        )

    if x < xp.min():
        x0, x1 = xp[0], xp[1]
        y0, y1 = fp[0], fp[1]
    else:
        x0, x1 = xp[-2], xp[-1]
        y0, y1 = fp[-2], fp[-1]

    slope = (y1 - y0) / (x1 - x0)

    return float(y0 + slope * (x - x0)), "extrapolated"


def estimate_smail_params_from_anchors(
    anchor_rows: list[dict[str, Any]],
    target_neff: float,
    *,
    allow_extrapolate: bool,
) -> dict[str, Any]:
    """Estimate Smail parameters for any target n_eff from anchor fits."""
    good_rows = sorted(
        [row for row in anchor_rows if row["ok"]],
        key=lambda row: row["neff"],
    )

    if len(good_rows) < 2:
        raise ValueError("At least two successful anchor fits are needed.")

    neff_grid = np.array([row["neff"] for row in good_rows], dtype=float)
    alpha_grid = np.array([row["alpha"] for row in good_rows], dtype=float)
    beta_grid = np.array([row["beta"] for row in good_rows], dtype=float)
    z0_grid = np.array([row["z0"] for row in good_rows], dtype=float)

    alpha, alpha_method = linear_interp_or_extrap(
        target_neff,
        neff_grid,
        alpha_grid,
        allow_extrapolate=allow_extrapolate,
    )
    beta, beta_method = linear_interp_or_extrap(
        target_neff,
        neff_grid,
        beta_grid,
        allow_extrapolate=allow_extrapolate,
    )
    z0, z0_method = linear_interp_or_extrap(
        target_neff,
        neff_grid,
        z0_grid,
        allow_extrapolate=allow_extrapolate,
    )

    methods = {alpha_method, beta_method, z0_method}
    method = "extrapolated" if "extrapolated" in methods else "interpolated"

    return {
        "ok": True,
        "neff": float(target_neff),
        "n": method,
        "alpha": alpha,
        "beta": beta,
        "z0": z0,
        "fun": np.nan,
        "path": method,
        "method": method,
    }


def get_smail_params_for_neff(
    base_dir: Path,
    anchor_rows: list[dict[str, Any]],
    target_neff: float,
    *,
    alpha: float,
    z_max: float,
    min_n: int,
    allow_extrapolate: bool,
    prefer_exact_file: bool,
) -> dict[str, Any]:
    """Return Smail parameters for any target n_eff.

    If prefer_exact_file is True and roman_n_eff{target}.nz exists, the target
    file is fitted directly. Otherwise the result is estimated from the anchor
    fits by interpolation or extrapolation.
    """
    if prefer_exact_file:
        direct_fit = fit_one_neff(
            base_dir,
            target_neff,
            alpha=alpha,
            z_max=z_max,
            min_n=min_n,
        )

        if direct_fit["ok"]:
            return direct_fit

    return estimate_smail_params_from_anchors(
        anchor_rows,
        target_neff,
        allow_extrapolate=allow_extrapolate,
    )


def get_wenzl22_smail_params(
    base_dir: Path,
    anchor_rows: list[dict[str, Any]],
    *,
    alpha: float,
    z_max: float,
    min_n: int,
    allow_extrapolate: bool,
    prefer_exact_file: bool,
) -> list[dict[str, Any]]:
    """Return Smail parameters for Wenzl22 Roman HLS/wide lens/source cases."""
    wenzl22_rows: list[dict[str, Any]] = []

    for scenario, values in WENZL22_SCENARIOS.items():
        for role in ("source", "lens"):
            target_neff = float(values[f"{role}_neff"])

            row = get_smail_params_for_neff(
                base_dir,
                anchor_rows,
                target_neff,
                alpha=alpha,
                z_max=z_max,
                min_n=min_n,
                allow_extrapolate=allow_extrapolate,
                prefer_exact_file=prefer_exact_file,
            )

            wenzl22_rows.append(
                {
                    **row,
                    "scenario": scenario,
                    "role": role,
                    "photoz_scatter": values["photoz_scatter"],
                    "survey_area_deg2": values["survey_area_deg2"],
                    "n_tomo_bins": values["n_tomo_bins"],
                    "binning_scheme": values["binning_scheme"],
                    "reference": "Wenzl22",
                }
            )

    return wenzl22_rows


def print_yaml_block(row: dict[str, Any]) -> None:
    """Print one generic copy-paste YAML block."""
    print(f"\n# n_eff = {row['neff']}; method = {row.get('method', row['n'])}")
    print("nz:")
    print("  model: smail")
    print("  params:")
    print(f"    alpha: {row['alpha']:.2f}")
    print(f"    beta: {row['beta']:.2f}")
    print(f"    z0: {row['z0']:.2f}")


def print_wenzl22_yaml_block(row: dict[str, Any]) -> None:
    """Print one Wenzl22 scenario YAML block."""
    print(f"\n# {row['scenario']} {row['role']}")
    print("# reference = Wenzl22")
    print(f"# survey area = {row['survey_area_deg2']:.0f} deg^2")
    print(f"# n_eff = {row['neff']}; method = {row.get('method', row['n'])}")
    print(f"# photo-z scatter sigma_z = {row['photoz_scatter']}")
    print(f"# tomography = {row['n_tomo_bins']} {row['binning_scheme']} bins")
    print(f"{row['role']}:")
    print("  n_gal_arcmin2:", f"{row['neff']:.1f}")
    print("  nz:")
    print("    model: smail")
    print("    params:")
    print(f"      alpha: {row['alpha']:.2f}")
    print(f"      beta: {row['beta']:.2f}")
    print(f"      z0: {row['z0']:.2f}")
    print("  bins:")
    print(f"    scheme: {row['binning_scheme']}")
    print(f"    n_bins: {row['n_tomo_bins']}")
    print("  photoz:")
    print(f"    scatter: {row['photoz_scatter']:.3f}")


def main() -> None:
    """Fit anchor Smail parameters and estimate targets for arbitrary n_eff."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Fit fixed-alpha Smail parameters from Roman n(z) files and "
            "interpolate/extrapolate parameters for arbitrary n_eff values."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=script_dir / "data",
        help="Directory containing roman_n_eff*.nz files.",
    )
    parser.add_argument(
        "--anchor-neff",
        "--neff",
        dest="anchor_neff",
        type=float,
        nargs="+",
        default=[20, 30, 40, 50],
        help="n_eff files to fit as interpolation/extrapolation anchors.",
    )
    parser.add_argument(
        "--target-neff",
        "--interp-neff",
        dest="target_neff",
        type=float,
        nargs="*",
        default=[],
        help="Any desired n_eff values to estimate from anchor fits.",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument(
        "--wenzl22",
        action="store_true",
        help="Print Wenzl22 Roman HLS/wide lens/source Smail parameters.",
    )
    parser.add_argument(
        "--paper-scenarios",
        action="store_true",
        help="Alias for --wenzl22.",
    )
    parser.add_argument(
        "--no-extrapolate",
        action="store_true",
        help="Disable extrapolation outside the anchor n_eff range.",
    )
    parser.add_argument(
        "--no-exact-target-fit",
        action="store_true",
        help="Always estimate target n_eff values from anchors, even if a file exists.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for saving results as CSV.",
    )

    args = parser.parse_args()

    allow_extrapolate = not args.no_extrapolate
    prefer_exact_file = not args.no_exact_target_fit
    use_wenzl22 = args.wenzl22 or args.paper_scenarios

    anchor_rows = [
        fit_one_neff(
            args.base_dir,
            neff,
            alpha=args.alpha,
            z_max=args.z_max,
            min_n=args.min_n,
        )
        for neff in args.anchor_neff
    ]

    target_rows = [
        get_smail_params_for_neff(
            args.base_dir,
            anchor_rows,
            target_neff,
            alpha=args.alpha,
            z_max=args.z_max,
            min_n=args.min_n,
            allow_extrapolate=allow_extrapolate,
            prefer_exact_file=prefer_exact_file,
        )
        for target_neff in args.target_neff
    ]

    wenzl22_rows: list[dict[str, Any]] = []

    if use_wenzl22:
        wenzl22_rows = get_wenzl22_smail_params(
            args.base_dir,
            anchor_rows,
            alpha=args.alpha,
            z_max=args.z_max,
            min_n=args.min_n,
            allow_extrapolate=allow_extrapolate,
            prefer_exact_file=prefer_exact_file,
        )

    all_rows = anchor_rows + target_rows + wenzl22_rows
    good_rows = [row for row in all_rows if row["ok"]]
    bad_rows = [row for row in all_rows if not row["ok"]]

    if good_rows:
        table = pd.DataFrame(good_rows)
        cols = [
            col
            for col in [
                "reference",
                "scenario",
                "role",
                "survey_area_deg2",
                "neff",
                "n",
                "alpha",
                "beta",
                "z0",
                "photoz_scatter",
                "n_tomo_bins",
                "binning_scheme",
                "method",
                "fun",
                "path",
            ]
            if col in table.columns
        ]

        print("\nSmail parameters:")
        print(table[cols].to_string(index=False))

        anchor_good_rows = [row for row in anchor_rows if row["ok"]]
        target_good_rows = [row for row in target_rows if row["ok"]]

        if anchor_good_rows:
            print("\nAnchor YAML blocks:")
            for row in anchor_good_rows:
                print_yaml_block(row)

        if target_good_rows:
            print("\nTarget YAML blocks:")
            for row in target_good_rows:
                print_yaml_block(row)

        if wenzl22_rows:
            print("\nWenzl22 scenario YAML blocks:")
            for row in wenzl22_rows:
                print_wenzl22_yaml_block(row)

        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            table[cols].to_csv(args.output_csv, index=False)
            print(f"\nSaved CSV: {args.output_csv}")

    if bad_rows:
        print("\nFailed anchor files:")
        print(pd.DataFrame(bad_rows).to_string(index=False))


if __name__ == "__main__":
    main()
