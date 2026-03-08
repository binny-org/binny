from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from binny.api.nz_tomography import NZTomography

FPS = 14
PAUSE_FRAMES = 2
TRANSITION_FRAMES = 10
FIGSIZE = (8.8, 7.2)

DEFAULT_XLIM = (0.0, 0.6)
FILL_ALPHA = 0.65
LINEWIDTH = 5.0


def build_binny_curves():
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

    b_sum = np.zeros_like(z)
    for k in keys:
        b_sum += np.asarray(bins[k])

    eps = 1e-30
    shrink = 0.8
    scaled = []

    for k in keys:
        b = np.asarray(bins[k])
        frac = b / np.maximum(b_sum, eps)
        scaled.append(shrink * nz * frac)

    return z, scaled


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


def plot_logo(ax, z, curves, colors, n_full, next_scale, ylim):
    ax.cla()

    # Fully drawn curves
    for i in range(n_full):
        curve = curves[i]
        color = colors[i]

        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=FILL_ALPHA,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=LINEWIDTH,
        )

    # Incoming curve grows from the baseline
    if n_full < len(curves) and next_scale > 0.0:
        curve = next_scale * curves[n_full]
        color = colors[n_full]

        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=FILL_ALPHA,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=LINEWIDTH,
        )

    ax.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=LINEWIDTH,
    )

    ax.set_xlim(*DEFAULT_XLIM)
    ax.set_ylim(*ylim)
    ax.axis("off")


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "binny_logo.gif"

z, curves = build_binny_curves()
n_bins = len(curves)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

ymax = max(np.max(c) for c in curves)
ylim = (-0.01, 1.15 * ymax)

# Each frame stores:
#   n_full = number of fully drawn curves
#   next_scale = growth factor for the next curve
timeline = []

timeline.extend([(1, 0.0)] * PAUSE_FRAMES)

for i in range(1, n_bins):
    for j in range(TRANSITION_FRAMES):
        t = (j + 1) / TRANSITION_FRAMES
        s = smootherstep(t)
        timeline.append((i, s))
    timeline.extend([(i + 1, 0.0)] * PAUSE_FRAMES)

fig, ax = plt.subplots(figsize=FIGSIZE)

fig.subplots_adjust(
    left=0.04,
    right=0.96,
    top=0.96,
    bottom=0.04,
)


def update(frame):
    n_full, next_scale = timeline[frame]

    plot_logo(
        ax,
        z,
        curves,
        colors,
        n_full,
        next_scale,
        ylim,
    )

    return []


anim = FuncAnimation(
    fig,
    update,
    frames=len(timeline),
    interval=1000 / FPS,
    blit=False,
)

anim.save(
    OUTFILE,
    writer=PillowWriter(fps=FPS),
)

plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
