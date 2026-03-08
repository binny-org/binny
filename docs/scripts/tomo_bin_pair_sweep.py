from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from binny import NZTomography


def ordered_curves(bin_dict):
    keys = sorted(bin_dict.keys())
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def pair_list(n_bins):
    return [(i, j) for i in range(n_bins) for j in range(i, n_bins)]


def square_triangles(x, y, size=0.42):
    x0, x1 = x - size, x + size
    y0, y1 = y - size, y + size

    tri1 = [(x0, y0), (x1, y0), (x0, y1)]
    tri2 = [(x1, y1), (x1, y0), (x0, y1)]
    border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    diag = [(x1, y0), (x0, y1)]

    return tri1, tri2, border, diag


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_bin_pair_sweep.gif"

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
curves = ordered_curves(result.bins)
n_bins = len(curves)
pairs = pair_list(n_bins)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.0, 5.0),
    gridspec_kw={"width_ratios": [1.8, 1.0]},
)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 17
lw = 2.0

# Left panel
ax = axes[0]
ax.set_title("Tomographic bin pairs", fontsize=title_fs)
ax.set_xlim(0.0, 2.0)
ax.set_ylim(0.0, 1.08 * max(np.max(c) for c in curves))
ax.set_xlabel("Redshift $z$", fontsize=label_fs)
ax.set_ylabel(r"Normalized $n_i(z)$", fontsize=label_fs)

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

fills = []
for i, color in enumerate(colors):
    fill = ax.fill_between(
        z,
        0.0,
        curves[i],
        color=color,
        alpha=0.18,
        linewidth=0.0,
        zorder=5,
    )
    fills.append(fill)

    ax.plot(
        z,
        curves[i],
        color="k",
        linewidth=1.4,
        alpha=0.35,
        zorder=10,
    )

highlight_lines = []
for _ in range(n_bins):
    (line,) = ax.plot([], [], color="k", linewidth=2.4, zorder=20)
    highlight_lines.append(line)


# Top line pieces: Pair: (i, j)
pair_prefix = ax.text(
    0.615,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
pair_i = ax.text(
    0.615,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
pair_comma = ax.text(
    0.615,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
pair_j = ax.text(
    0.615,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
pair_suffix = ax.text(
    0.615,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)

# Bottom line pieces
kind_first = ax.text(
    0.615,
    0.84,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
kind_dash = ax.text(
    0.615,
    0.84,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)
kind_second = ax.text(
    0.615,
    0.84,
    "",
    transform=ax.transAxes,
    fontsize=annot_fs,
    ha="left",
    va="top",
    color="k",
    zorder=41,
)

fig.canvas.draw()


def place_text_sequence(ax, y, pieces, x0=0.615):
    renderer = fig.canvas.get_renderer()
    ax_width_px = ax.get_window_extent(renderer=renderer).width

    x = x0
    for txt, content, color in pieces:
        txt.set_position((x, y))
        txt.set_text(content)
        txt.set_color(color)
        bb = txt.get_window_extent(renderer=renderer)
        x += bb.width / ax_width_px


# Right panel
axm = axes[1]
axm.set_title(r"Observable indices $C_\ell^{ij}$", fontsize=title_fs)

axm.set_xlim(-0.5, n_bins - 0.5)
axm.set_ylim(n_bins - 0.5, -0.5)

axm.set_xticks(range(n_bins))
axm.set_yticks(range(n_bins))

axm.set_xticklabels([str(i + 1) for i in range(n_bins)], fontsize=tick_fs)
axm.set_yticklabels([str(i + 1) for i in range(n_bins)], fontsize=tick_fs)

axm.set_xlabel("Bin $j$", fontsize=label_fs)
axm.set_ylabel("Bin $i$", fontsize=label_fs)

for side in ["left", "right", "top", "bottom"]:
    axm.spines[side].set_visible(True)
    axm.spines[side].set_linewidth(lw)

axm.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
    width=lw,
    length=6,
    labelsize=tick_fs,
)

axm.grid(False)

matrix = np.full((n_bins, n_bins), np.nan)
for i in range(n_bins):
    for j in range(i, n_bins):
        matrix[i, j] = 0.3

axm.imshow(
    matrix,
    vmin=0.0,
    vmax=1.0,
    cmap="Greys",
    interpolation="nearest",
)

for i in range(n_bins + 1):
    axm.axhline(i - 0.5, color="k", linewidth=1.2)
    axm.axvline(i - 0.5, color="k", linewidth=1.2)

tri1, tri2, border, diag = square_triangles(0, 0, size=0.42)

marker_tri1 = Polygon(
    tri1,
    closed=True,
    facecolor=colors[0],
    edgecolor="none",
    alpha=0.6,
    zorder=30,
    visible=False,
)

marker_tri2 = Polygon(
    tri2,
    closed=True,
    facecolor=colors[0],
    edgecolor="none",
    alpha=0.6,
    zorder=31,
    visible=False,
)

marker_border = Polygon(
    border,
    closed=True,
    facecolor="none",
    edgecolor="k",
    linewidth=2.4,
    zorder=32,
    visible=False,
)

(marker_diag,) = axm.plot(
    [diag[0][0], diag[1][0]],
    [diag[0][1], diag[1][1]],
    color="k",
    linewidth=1.8,
    zorder=33,
    visible=False,
)

axm.add_patch(marker_tri1)
axm.add_patch(marker_tri2)
axm.add_patch(marker_border)

frames_per_pair = 12
n_frames = frames_per_pair * len(pairs)


def init():
    for line in highlight_lines:
        line.set_data([], [])

    for fill in fills:
        fill.set_alpha(0.18)

    marker_tri1.set_visible(False)
    marker_tri2.set_visible(False)
    marker_border.set_visible(False)
    marker_diag.set_visible(False)

    for txt in [
        pair_prefix,
        pair_i,
        pair_comma,
        pair_j,
        pair_suffix,
        kind_first,
        kind_dash,
        kind_second,
    ]:
        txt.set_text("")

    return (
        *highlight_lines,
        marker_tri1,
        marker_tri2,
        marker_border,
        marker_diag,
        pair_prefix,
        pair_i,
        pair_comma,
        pair_j,
        pair_suffix,
        kind_first,
        kind_dash,
        kind_second,
    )


def update(frame):
    pair_idx = min(frame // frames_per_pair, len(pairs) - 1)
    i, j = pairs[pair_idx]

    for b in range(n_bins):
        if b == i or b == j:
            highlight_lines[b].set_data(z, curves[b])
            fills[b].set_alpha(0.6)
        else:
            highlight_lines[b].set_data([], [])
            fills[b].set_alpha(0.18)

    tri1, tri2, border, diag = square_triangles(j, i, size=0.42)

    marker_tri1.set_xy(tri1)
    marker_tri2.set_xy(tri2)
    marker_border.set_xy(border)

    marker_tri1.set_facecolor(colors[i])
    marker_tri2.set_facecolor(colors[j])
    marker_tri1.set_alpha(0.6)
    marker_tri2.set_alpha(0.6)

    marker_diag.set_data(
        [diag[0][0], diag[1][0]],
        [diag[0][1], diag[1][1]],
    )

    marker_tri1.set_visible(True)
    marker_tri2.set_visible(True)
    marker_border.set_visible(True)
    marker_diag.set_visible(True)

    place_text_sequence(
        ax,
        0.94,
        [
            (pair_prefix, "Pair: ", "k"),
            (pair_i, f"{i + 1}", colors[i]),
            (pair_comma, ", ", "k"),
            (pair_j, f"{j + 1}", colors[j]),
            (pair_suffix, "", "k"),
        ],
    )

    if i == j:
        first_word = "Auto"
        second_word = "correlation"
        second_color = colors[i]
    else:
        first_word = "Cross"
        second_word = "correlation"
        second_color = colors[j]

    place_text_sequence(
        ax,
        0.84,
        [
            (kind_first, first_word, colors[i]),
            (kind_dash, "-", "k"),
            (kind_second, second_word, second_color),
        ],
    )

    return (
        *highlight_lines,
        marker_tri1,
        marker_tri2,
        marker_border,
        marker_diag,
        pair_prefix,
        pair_i,
        pair_comma,
        pair_j,
        pair_suffix,
        kind_first,
        kind_dash,
        kind_second,
    )


plt.tight_layout()
fig.canvas.draw()

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=140,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=8))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
