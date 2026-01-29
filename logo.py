"""
binny/logo.py

Logo plot WITHOUT YAML using NZTomography.build_bins (arrays path).

Trick:
- Plot *envelope-scaled* versions of the bins so they sit inside the parent Smail:
    b_scaled(z) = fraction_of_smail * nz(z) * (b(z) / sum_i b_i(z))
"""

from __future__ import annotations

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography  # adjust if needed


def main() -> None:
    # -------------------------
    # 1) Parent n(z)
    # -------------------------
    z = np.linspace(0, 3, 1000)
    nz = NZTomography.nz_model("smail", z, z0=0.26, alpha=2.0, beta=0.94, normalize=True)

    # -------------------------
    # 2) Tomo spec
    # -------------------------
    tomo_spec = {
        "kind": "photoz",
        "nz": {"model": "arrays"},  # harmless stub; parser may want it
        "bins": {
            "scheme": "equipopulated",
            "n_bins": 3,
        },
        "uncertainties": {"scatter_scale": 0.1, "mean_offset": 0.01},
    }

    # -------------------------
    # 3) Build bins from arrays
    # -------------------------
    t = NZTomography()
    payload = t.build_bins(z=z, nz=nz, tomo_spec=tomo_spec, include_tomo_metadata=False)
    bins = payload["bins"]

    # -------------------------
    # 4) Plot (logo style)
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    keys = sorted(bins.keys())
    colors = cmr.take_cmap_colors(
        "cmr.neon",
        len(keys),
        cmap_range=(0.25, 1.0),
        return_fmt="hex",
    )

    # compute total at each z (for p_i = b_i / sum b_i)
    b_sum = np.zeros_like(z, dtype=float)
    for k in keys:
        b_sum += np.asarray(bins[k], dtype=float)

    eps = 1e-30
    shrink = 0.85
    fill_alpha = 0.4

    for k, color in zip(keys, colors, strict=True):
        b = np.asarray(bins[k], dtype=float)
        p = b / np.maximum(b_sum, eps)
        b_under = shrink * nz * p

        ax.fill_between(z, 0.0, b_under, color=color, alpha=fill_alpha, linewidth=0)
        ax.plot(z, b_under, color=color, linewidth=2)

    ax.text(
        0.85,
        -0.02,
        "binny",
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        ha="center",
        va="top",
        family="DejaVu Sans",
    )

    ax.axis("off")
    fig.tight_layout()

    fig.savefig("binny_logo.svg", transparent=True)
    fig.savefig("binny_logo.png", dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
