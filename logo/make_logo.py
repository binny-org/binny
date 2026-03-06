"""Binny logo built from tomographic redshift bins."""

from __future__ import annotations

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography

OUTDIR = Path(__file__).resolve().parent
SWEEP_OUTDIR = OUTDIR / "cmap_sweep"
DOCS_ASSETS_OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "_static" / "assets"

DEFAULT_CMAP = "viridis"
DEFAULT_CMAP_RANGE = (0.0, 1.0)


def _build_binny_curves(
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Build the z grid, scaled tomographic curves, and colors."""
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
            "n_bins": 4,
        },
        "uncertainties": {
            "scatter_scale": 0.03,
            "mean_offset": 0.0,
        },
    }

    result = NZTomography().build_bins(
        z=z,
        nz=nz,
        tomo_spec=tomo_spec,
        include_tomo_metadata=False,
    )
    bins = result.bins
    keys = sorted(bins.keys())

    colors = cmr.take_cmap_colors(
        cmap,
        len(keys),
        cmap_range=cmap_range,
        return_fmt="hex",
    )

    b_sum = np.zeros_like(z, dtype=float)
    for key in keys:
        b_sum += np.asarray(bins[key], dtype=float)

    eps = 1e-30
    shrink = 0.8
    scaled_bins: list[np.ndarray] = []

    for key in keys:
        b = np.asarray(bins[key], dtype=float)
        frac = b / np.maximum(b_sum, eps)
        scaled_bins.append(shrink * nz * frac)

    return z, scaled_bins, colors


def _render_logo(
    *,
    figsize: tuple[float, float],
    fill_alpha: float,
    linewidth: float,
    xlim: tuple[float, float],
    ylim_scale: float,
    pad: float,
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a Binny logo with configurable styling."""
    z, scaled_bins, colors = _build_binny_curves(cmap=cmap, cmap_range=cmap_range)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (curve, color) in enumerate(zip(scaled_bins, colors, strict=True)):
        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=10 + i,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=linewidth,
            zorder=10 + i,
        )

    ax.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=linewidth,
        zorder=1000,
        solid_capstyle="butt",
    )

    ymax = max(float(np.max(curve)) for curve in scaled_bins)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.01, ymax * ylim_scale)
    ax.axis("off")
    fig.tight_layout(pad=pad)

    return fig, ax


def make_binny_logo(
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[plt.Figure, plt.Axes]:
    """Construct the standard Binny logo figure and axes."""
    return _render_logo(
        figsize=(5, 5),
        fill_alpha=0.65,
        linewidth=5.0,
        xlim=(0.0, 0.6),
        ylim_scale=1.15,
        pad=0.1,
        cmap=cmap,
        cmap_range=cmap_range,
    )


def make_binny_favicon(
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[plt.Figure, plt.Axes]:
    """Construct a square favicon version of the Binny logo."""
    return _render_logo(
        figsize=(4, 4),
        fill_alpha=0.85,
        linewidth=3.0,
        xlim=(0.1, 0.42),
        ylim_scale=1.05,
        pad=0.02,
        cmap=cmap,
        cmap_range=cmap_range,
    )


def _save_figure(
    fig: plt.Figure,
    outputs: list[Path],
    *,
    transparent: bool = True,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    dpi: int | None = None,
) -> None:
    """Save a figure to one or more output files and close it."""
    for path in outputs:
        save_kwargs = {
            "transparent": transparent,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
        }
        if dpi is not None:
            save_kwargs["dpi"] = dpi

        fig.savefig(path, **save_kwargs)

    plt.close(fig)


def save_logo(
    outbase: Path,
    cmap: str,
    cmap_range: tuple[float, float],
) -> None:
    """Build and save one logo to SVG, PNG, and PDF."""
    fig, _ = make_binny_logo(cmap=cmap, cmap_range=cmap_range)
    _save_figure(
        fig,
        [
            outbase.with_suffix(".svg"),
            outbase.with_suffix(".png"),
            outbase.with_suffix(".pdf"),
        ],
    )


def save_favicon(
    outfile: Path,
    cmap: str,
    cmap_range: tuple[float, float],
) -> None:
    """Build and save a favicon PNG."""
    fig, _ = make_binny_favicon(cmap=cmap, cmap_range=cmap_range)
    _save_figure(fig, [outfile], dpi=256)


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
    """Build the default logo, favicon, and a colormap sweep."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_OUTDIR.mkdir(parents=True, exist_ok=True)

    save_logo(
        outbase=DOCS_ASSETS_OUTDIR / "logo",
        cmap=DEFAULT_CMAP,
        cmap_range=DEFAULT_CMAP_RANGE,
    )

    save_favicon(
        outfile=DOCS_ASSETS_OUTDIR / "favicon.png",
        cmap=DEFAULT_CMAP,
        cmap_range=DEFAULT_CMAP_RANGE,
    )

    sweep_cmaps()

    fig, _ = make_binny_logo()
    plt.show()


if __name__ == "__main__":
    main()
