#!/usr/bin/env python3
"""
Plot photo-z tomography bins from shipped survey YAML.

- Uses NZTomography.build_survey_bins
- No overrides
- No touching systematics
- No manual config handling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography


def plot_bins(
    z: np.ndarray,
    bins: dict[int, np.ndarray],
    *,
    title: str,
    outfile: Path,
) -> None:
    fig, ax = plt.subplots()

    keys = sorted(bins.keys())
    colors = cmr.take_cmap_colors("cmr.neon", len(keys), cmap_range=(0.25, 1.0), return_fmt="hex")

    for k, color in zip(keys, colors, strict=True):
        ax.plot(z, bins[k], color=color, lw=1.5, label=f"bin {k}")

    ax.set_xlabel("z")
    ax.set_ylabel(r"$n_i(z)$")
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", type=str, default="lsst")
    parser.add_argument("--role", type=str, default="source")
    parser.add_argument("--year", type=str, default="1")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    t = NZTomography()

    payload = t.build_survey_bins(
        survey=args.survey,
        role=args.role,
        year=args.year,
        include_survey_metadata=False,
        include_tomo_metadata=False,
        include_stats=False,
    )

    z = payload["z"]
    bins = payload["bins"]

    outfile = Path(args.outdir) / f"{args.survey}_{args.role}_Y{args.year}_photoz_bins.png"

    plot_bins(
        z,
        bins,
        title=f"{args.survey.upper()} {args.role} Y{args.year} photo-z bins",
        outfile=outfile,
    )

    print(f"Saved {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
