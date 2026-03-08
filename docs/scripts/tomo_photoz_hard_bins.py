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


def ordered_curves(bin_dict):
    keys = sorted(bin_dict.keys())
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def top_hat_window(z, zmin, zmax):
    return ((z >= zmin) & (z < zmax)).astype(float)


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_photoz_hard_bins.gif"

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

n_bins = 4
bin_range = (0.2, 1.2)
edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

# Top panel: hard-cut parent pieces
hard_curves = []
for i in range(n_bins):
    w = top_hat_window(z, edges[i], edges[i + 1])
    hard_curves.append(nz * w)

# Bottom panel: one fixed photo-z bin construction
spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": n_bins,
        "range": bin_range,
    },
    "uncertainties": {
        "scatter_scale": 0.05,
    },
    "normalize_bins": True,
}

result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
photoz_curves = ordered_curves(result.bins)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

reveal_vals = np.linspace(z.min(), z.max(), 50)

ymax_parent = 1.08 * np.max(nz)
ymax_bins = 1.08 * max(np.max(c) for c in photoz_curves)

fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 15
lw = 2.0

titles = ["Parent distribution with hard cuts", "Photometric tomographic bins"]

for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlim(0.0, 2.0)

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

axes[0].set_ylabel(r"$n(z)$", fontsize=label_fs)
axes[1].set_ylabel(r"$n_i(z)$", fontsize=label_fs)
axes[1].set_xlabel("Redshift $z$", fontsize=label_fs)

axes[0].plot(z, nz, color="k", linewidth=lw, zorder=20)

fills_parent, fills_bins = [], []
lines_parent, lines_bins = [], []

for _i, color in enumerate(colors):
    fp = axes[0].fill_between(
        z,
        0.0,
        np.zeros_like(z),
        color=color,
        alpha=0.6,
        linewidth=0.0,
        zorder=10,
    )
    (lp,) = axes[0].plot([], [], color="k", linewidth=lw, zorder=25)
    fills_parent.append(fp)
    lines_parent.append(lp)

    fb = axes[1].fill_between(
        z,
        0.0,
        np.zeros_like(z),
        color=color,
        alpha=0.6,
        linewidth=0.0,
        zorder=10,
    )
    (lb,) = axes[1].plot([], [], color="k", linewidth=lw, zorder=20)
    fills_bins.append(fb)
    lines_bins.append(lb)

text_top = axes[0].text(
    0.60,
    0.93,
    "",
    transform=axes[0].transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
)

text_bottom = axes[1].text(
    0.5,
    0.93,
    "",
    transform=axes[1].transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
)


def init():
    global fills_parent, fills_bins

    zcut = reveal_vals[0]

    for i in range(n_bins):
        y_parent = np.where(z <= zcut, hard_curves[i], 0.0)
        fills_parent[i] = replace_fill(axes[0], fills_parent[i], z, y_parent, colors[i])
        lines_parent[i].set_data(z, y_parent)

        y_bin = np.where(z <= zcut, photoz_curves[i], 0.0)
        fills_bins[i] = replace_fill(axes[1], fills_bins[i], z, y_bin, colors[i])
        lines_bins[i].set_data(z, y_bin)

    text_top.set_text("Sharp, non-overlapping redshift intervals")
    text_bottom.set_text(r"Smooth, overlapping photo-z bins")
    return *lines_parent, *lines_bins, text_top, text_bottom


def update(i):
    global fills_parent, fills_bins

    zcut = reveal_vals[i]

    for b in range(n_bins):
        y_parent = np.where(z <= zcut, hard_curves[b], 0.0)
        fills_parent[b] = replace_fill(axes[0], fills_parent[b], z, y_parent, colors[b])
        lines_parent[b].set_data(z, y_parent)

        y_bin = np.where(z <= zcut, photoz_curves[b], 0.0)
        fills_bins[b] = replace_fill(axes[1], fills_bins[b], z, y_bin, colors[b])
        lines_bins[b].set_data(z, y_bin)

    text_top.set_text(rf"Hard bins over $z \in [{bin_range[0]:.1f},\, {bin_range[1]:.1f}]$")
    text_bottom.set_text(r"Smooth, overlapping photo-z bins")
    return *lines_parent, *lines_bins, text_top, text_bottom


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=len(reveal_vals),
    init_func=init,
    interval=120,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=10))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
