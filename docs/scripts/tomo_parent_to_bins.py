from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from binny import NZTomography


def replace_fill(ax, old_fill, x, y, color, alpha=0.6):
    old_fill.remove()
    return ax.fill_between(
        x,
        0.0,
        y,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=10,
    )


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_parent_to_bins.gif"

# Parent distribution
tomo = NZTomography()

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "scatter_scale": 0.05,
    },
    "normalize_bins": True,
}

result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)

keys = sorted(result.bins.keys())
bin_curves = [np.asarray(result.bins[k], dtype=float) for k in keys]
n_bins = len(bin_curves)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=False)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 17
lw = 2.0

ymax_parent = 1.08 * np.max(nz)
ymax_bins = 1.08 * max(np.max(c) for c in bin_curves)

for ax in axes:
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("Redshift $z$", fontsize=label_fs)

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=lw,
        length=6,
        labelsize=tick_fs,
    )
    ax.grid(False)
    ax.plot(z, np.zeros_like(z), color="k", linewidth=lw, zorder=100)

axes[0].set_ylim(0.0, ymax_parent)
axes[1].set_ylim(0.0, ymax_bins)

axes[0].set_title("Parent distribution", fontsize=title_fs)
axes[1].set_title("Tomographic bins", fontsize=title_fs)
axes[0].set_ylabel(r"Normalized $n(z)$", fontsize=label_fs)

axes[0].fill_between(z, 0.0, nz, color="0.82", alpha=0.9, linewidth=0.0, zorder=2)
axes[0].plot(z, nz, color="k", linewidth=lw, zorder=3)

fills = []
lines = []
labels = []

for i, color in enumerate(colors):
    fill = axes[1].fill_between(
        z,
        0.0,
        np.zeros_like(z),
        color=color,
        alpha=0.6,
        linewidth=0.0,
        zorder=10,
    )
    (line,) = axes[1].plot([], [], color="k", linewidth=lw, zorder=20)
    label = axes[1].text(
        0.8,
        0.94 - i * 0.10,
        rf"Bin {i + 1}",
        transform=axes[1].transAxes,
        fontsize=annot_fs,
        ha="left",
        va="top",
        alpha=0.6,
        color="black",
        bbox=dict(
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            alpha=0.6,
        ),
    )
    fills.append(fill)
    lines.append(line)
    labels.append(label)


n_reveal = 20
n_hold = 10
n_frames = n_reveal + n_hold


def init():
    global fills

    for i in range(n_bins):
        fills[i] = replace_fill(axes[1], fills[i], z, np.zeros_like(z), colors[i])
        lines[i].set_data([], [])
        labels[i].set_alpha(0.0)

    return *fills, *lines, *labels


def update(frame):
    global fills

    frac = 1.0 if frame >= n_reveal else (frame + 1) / n_reveal

    for i in range(n_bins):
        y = frac * bin_curves[i]
        lines[i].set_data(z, y)
        fills[i] = replace_fill(axes[1], fills[i], z, y, colors[i])
        labels[i].set_alpha(frac)

    return *fills, *lines, *labels


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=120,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=10))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
