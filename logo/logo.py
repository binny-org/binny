"""Binny logo built from tomographic redshift bins."""

from __future__ import annotations

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography

OUTDIR = Path(__file__).resolve().parent
SWEEP_OUTDIR = OUTDIR / "cmap_sweep"

DEFAULT_CMAP = "viridis"
DEFAULT_CMAP_RANGE = (0, 1)


def make_binny_logo(
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[plt.Figure, plt.Axes]:
    """Construct the Binny logo figure and axes."""
    z = np.linspace(0.0, 1.0, 1000)

    nz = NZTomography.nz_model(
        "smail",
        z,
        z0=0.22,
        alpha=2.5,
        beta=2.2,
        normalize=True,
    )

    tomo_spec = {
        "kind": "photoz",
        "nz": {"model": "arrays"},
        "bins": {
            "scheme": "equipopulated",
            "n_bins": 3,
        },
        "uncertainties": {
            "scatter_scale": 0.03,
            "mean_offset": 0.0,
        },
    }

    t = NZTomography()
    result = t.build_bins(
        z=z,
        nz=nz,
        tomo_spec=tomo_spec,
        include_tomo_metadata=False,
    )
    bins = result.bins

    fig, ax = plt.subplots(figsize=(10, 5))

    keys = sorted(bins.keys())
    colors = cmr.take_cmap_colors(
        cmap,
        len(keys),
        cmap_range=cmap_range,
        return_fmt="hex",
    )

    b_sum = np.zeros_like(z, dtype=float)
    for k in keys:
        b_sum += np.asarray(bins[k], dtype=float)

    eps = 1e-30
    shrink = 0.88
    fill_alpha = 0.65
    lw = 5

    for i, (k, color) in enumerate(zip(keys, colors, strict=True)):
        b = np.asarray(bins[k], dtype=float)
        frac = b / np.maximum(b_sum, eps)
        b_scaled = shrink * nz * frac

        ax.fill_between(
            z,
            0.0,
            b_scaled,
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=10 + i,
        )
        ax.plot(
            z,
            b_scaled,
            color="k",
            linewidth=lw,
            zorder=10 + i,
        )

    ax.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=lw,
        zorder=1000,
        solid_capstyle="butt",
    )

    ax.set_xlim(0.0, 0.6)
    ax.axis("off")
    fig.tight_layout(pad=0.1)

    return fig, ax


def save_logo(
    outbase: Path,
    cmap: str,
    cmap_range: tuple[float, float],
) -> None:
    """Build and save one logo to SVG and PNG."""
    fig, _ = make_binny_logo(cmap=cmap, cmap_range=cmap_range)

    suffix = [".svg", ".png", ".pdf"]

    for s in suffix:
        fig.savefig(outbase.with_suffix(s), transparent=True, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


def sweep_cmaps() -> None:
    """Save a sweep of requested colormaps."""
    SWEEP_OUTDIR.mkdir(parents=True, exist_ok=True)

    sweep_specs: list[tuple[str, str, tuple[float, float]]] = [
        ("cmr_pride", "cmr.pride", (0.2, 0.8)),
        ("cmr_eclipse", "cmr.eclipse", (0.2, 1.0)),
        ("cmr_ghostlight", "cmr.ghostlight", (0.2, 1.0)),
        ("cmr_neon", "cmr.neon", (0.2, 1.0)),
        ("cmr_tropical", "cmr.tropical", (0.0, 1.0)),
        ("cmr_torch", "cmr.torch", (0.2, 0.8)),
        ("viridis", "viridis", (0.0, 1.0)),
        ("plasma", "plasma", (0.0, 1.0)),
    ]

    for stem, cmap, cmap_range in sweep_specs:
        save_logo(
            outbase=SWEEP_OUTDIR / f"binny_logo_{stem}",
            cmap=cmap,
            cmap_range=cmap_range,
        )


def main() -> None:
    """Build the default logo and a colormap sweep."""
    OUTDIR.mkdir(parents=True, exist_ok=True)

    save_logo(
        outbase=OUTDIR / "binny_logo",
        cmap=DEFAULT_CMAP,
        cmap_range=DEFAULT_CMAP_RANGE,
    )

    sweep_cmaps()

    fig, _ = make_binny_logo()
    plt.show()


if __name__ == "__main__":
    main()
