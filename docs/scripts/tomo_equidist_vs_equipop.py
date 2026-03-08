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


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_equidist_vs_equipop.gif"

# Parent distribution and photo-z specs
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

common_uncertainties = {"scatter_scale": 0.05}

equidistant_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equipopulated_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equidistant_spec)
equipopulated_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equipopulated_spec)

curves_eqd = ordered_curves(equidistant_result.bins)
curves_eqp = ordered_curves(equipopulated_result.bins)

n_bins = len(curves_eqd)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

ymax = 1.08 * max(
    max(np.max(c) for c in curves_eqd),
    max(np.max(c) for c in curves_eqp),
)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 15
lw = 2.0

titles = ["Equidistant bins", "Equipopulated bins"]

for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, ymax)
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

axes[0].set_ylabel(r"Normalized $n_i(z)$", fontsize=label_fs)

fills_left, fills_right = [], []
lines_left, lines_right = [], []

for _i, color in enumerate(colors):
    f1 = axes[0].fill_between(
        z, 0.0, np.zeros_like(z), color=color, alpha=0.6, linewidth=0.0, zorder=10
    )
    (l1,) = axes[0].plot([], [], color="k", linewidth=lw, zorder=20)
    fills_left.append(f1)
    lines_left.append(l1)

    f2 = axes[1].fill_between(
        z, 0.0, np.zeros_like(z), color=color, alpha=0.6, linewidth=0.0, zorder=10
    )
    (l2,) = axes[1].plot([], [], color="k", linewidth=lw, zorder=20)
    fills_right.append(f2)
    lines_right.append(l2)

text_left = axes[0].text(
    0.5,
    0.94,
    "Equal redshift width",
    transform=axes[0].transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
)
text_right = axes[1].text(
    0.5,
    0.94,
    "Equal galaxy counts",
    transform=axes[1].transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
)

build_frames = 50
pause_seconds = 3
fps = 10
pause_frames = pause_seconds * fps
total_frames = build_frames + pause_frames


def init():
    global fills_left, fills_right

    for i in range(n_bins):
        fills_left[i] = replace_fill(axes[0], fills_left[i], z, np.zeros_like(z), colors[i])
        fills_right[i] = replace_fill(axes[1], fills_right[i], z, np.zeros_like(z), colors[i])
        lines_left[i].set_data([], [])
        lines_right[i].set_data([], [])

    return (*lines_left, *lines_right, text_left, text_right)


def update(i):
    global fills_left, fills_right

    if i < build_frames:
        frac = (i + 1) / build_frames
    else:
        frac = 1.0

    for b in range(n_bins):
        y_left = frac * curves_eqd[b]
        y_right = frac * curves_eqp[b]

        fills_left[b] = replace_fill(axes[0], fills_left[b], z, y_left, colors[b])
        fills_right[b] = replace_fill(axes[1], fills_right[b], z, y_right, colors[b])

        lines_left[b].set_data(z, y_left)
        lines_right[b].set_data(z, y_right)

    return (*lines_left, *lines_right, text_left, text_right)


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    init_func=init,
    interval=1000 / fps,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=fps))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
