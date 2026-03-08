from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from binny import NZTomography


def replace_fill(ax, old_fills, x, ys, colors, alpha=0.6):
    for fill in old_fills:
        fill.remove()

    new_fills = []
    for y, color in zip(ys, colors, strict=True):
        new_fills.append(
            ax.fill_between(
                x,
                0.0,
                y,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                zorder=10,
            )
        )
    return new_fills


def top_hat_window(z, zmin, zmax):
    return ((z >= zmin) & (z < zmax)).astype(float)


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_window_function_product.gif"

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
edges = np.linspace(0.2, 1.2, n_bins + 1)

windows = []
products = []

for i in range(n_bins):
    w = top_hat_window(z, edges[i], edges[i + 1])
    p = nz * w
    windows.append(w)
    products.append(p)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.8), sharex=True)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 15
lw = 2.0

titles = [r"Parent $n(z)$", r"Window $W_i(z)$", r"Product $n(z)W_i(z)$"]

for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title, fontsize=title_fs)
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

axes[0].set_ylim(0.0, 1.08 * np.max(nz))
axes[1].set_ylim(0.0, 1.15)
axes[2].set_ylim(0.0, 1.08 * np.max(nz))

axes[0].set_ylabel(r"$n(z)$", fontsize=label_fs)
axes[1].set_ylabel(r"$W_i(z)$", fontsize=label_fs)
axes[2].set_ylabel(r"$n_i(z)$", fontsize=label_fs)

axes[0].fill_between(z, 0.0, nz, color="0.84", alpha=0.9, linewidth=0.0, zorder=2)
axes[0].plot(z, nz, color="k", linewidth=lw, zorder=3)

fill_window = [
    axes[1].fill_between(z, 0.0, np.zeros_like(z), color=color, alpha=0.6, linewidth=0.0, zorder=10)
    for color in colors
]
fill_product = [
    axes[2].fill_between(z, 0.0, np.zeros_like(z), color=color, alpha=0.6, linewidth=0.0, zorder=10)
    for color in colors
]

line_windows = [axes[1].plot([], [], color="k", linewidth=lw, zorder=20)[0] for _ in range(n_bins)]
line_products = [axes[2].plot([], [], color="k", linewidth=lw, zorder=20)[0] for _ in range(n_bins)]

text_idx = axes[1].text(
    0.04,
    0.93,
    "All bins",
    transform=axes[1].transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
)

n_frames = 24


def init():
    global fill_window, fill_product

    zero_curves = [np.zeros_like(z) for _ in range(n_bins)]
    fill_window = replace_fill(axes[1], fill_window, z, zero_curves, colors)
    fill_product = replace_fill(axes[2], fill_product, z, zero_curves, colors)

    for line in line_windows:
        line.set_data(z, np.zeros_like(z))
    for line in line_products:
        line.set_data(z, np.zeros_like(z))

    return (*line_windows, *line_products, text_idx)


def update(frame):
    global fill_window, fill_product

    frac = (frame + 1) / n_frames

    current_windows = [frac * w for w in windows]
    current_products = [frac * p for p in products]

    fill_window = replace_fill(axes[1], fill_window, z, current_windows, colors)
    fill_product = replace_fill(axes[2], fill_product, z, current_products, colors)

    for line, y in zip(line_windows, current_windows, strict=True):
        line.set_data(z, y)

    for line, y in zip(line_products, current_products, strict=True):
        line.set_data(z, y)

    return (*line_windows, *line_products, text_idx)


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=130,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=8))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
