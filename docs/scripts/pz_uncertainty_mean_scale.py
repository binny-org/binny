from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from binny import NZTomography

DEFAULT_FONTSIZE = 19
plt.rcParams.update(
    {
        "font.size": DEFAULT_FONTSIZE,
        "axes.titlesize": DEFAULT_FONTSIZE,
        "axes.labelsize": DEFAULT_FONTSIZE,
        "xtick.labelsize": DEFAULT_FONTSIZE,
        "ytick.labelsize": DEFAULT_FONTSIZE,
        "legend.fontsize": DEFAULT_FONTSIZE,
        "figure.titlesize": DEFAULT_FONTSIZE,
    }
)

FPS = 10
FIGSIZE = (6.8, 6.2)


def ordered_curves(bin_dict):
    keys = sorted(bin_dict.keys())
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def make_pingpong_indices(n):
    forward = list(range(n))
    backward = list(range(n - 2, 0, -1))
    return forward + backward


def fill_vertices(x, y):
    return np.vstack(
        [
            [x[0], 0.0],
            np.column_stack([x, y]),
            [x[-1], 0.0],
        ]
    )


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "pz_uncertainty_mean_scale.gif"

tomo = NZTomography()

# Slightly fewer samples for faster drawing and smaller GIF
z = np.linspace(0.0, 2.0, 180)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

n_bins = 3
bin_range = (0.2, 1.1)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

baseline_scatter = 0.03
baseline_mean_scale = 1.0

# Fewer unique states -> much faster perceived animation
mean_scale_vals = np.linspace(1.0, 1.22, 16)

spec_fixed = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": n_bins,
        "range": bin_range,
    },
    "uncertainties": {
        "scatter_scale": baseline_scatter,
        "mean_scale": baseline_mean_scale,
        "mean_offset": 0.0,
        "outlier_frac": 0.0,
    },
    "normalize_bins": True,
}

result_fixed = tomo.build_bins(z=z, nz=nz, tomo_spec=spec_fixed)
fixed_curves = ordered_curves(result_fixed.bins)

all_curves = []
for mean_scale in mean_scale_vals:
    spec = {
        "kind": "photoz",
        "bins": {
            "scheme": "equidistant",
            "n_bins": n_bins,
            "range": bin_range,
        },
        "uncertainties": {
            "scatter_scale": [baseline_scatter, baseline_scatter, baseline_scatter],
            "mean_scale": [1.0, 1.0, float(mean_scale)],
            "mean_offset": [0.0, 0.0, 0.0],
            "outlier_frac": [0.0, 0.0, 0.0],
        },
        "normalize_bins": True,
    }

    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)

    frame_curves = [
        fixed_curves[0].copy(),
        fixed_curves[1].copy(),
        curves[2].copy(),
    ]
    all_curves.append(frame_curves)

frame_ids = make_pingpong_indices(len(mean_scale_vals))

ymax_bins = 1.08 * max(np.max(curve) for frame in all_curves for curve in frame)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

lw = 2.0

ax.set_title("Photo-z bins with increasing mean scaling")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(0.0, ymax_bins)

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
)

ax.grid(False)
ax.plot(z, np.zeros_like(z), color="k", linewidth=lw, zorder=100)

ax.set_ylabel(r"$n_i(z)$")
ax.set_xlabel("Redshift $z$")

fills_bins = []
lines_bins = []

zero_vertices = fill_vertices(z, np.zeros_like(z))

for color in colors:
    poly = Polygon(
        zero_vertices.copy(),
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=0.6,
        zorder=10,
        animated=True,
    )
    ax.add_patch(poly)
    fills_bins.append(poly)

    (line,) = ax.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=lw,
        zorder=20,
        animated=True,
    )
    lines_bins.append(line)

text_main = ax.text(
    0.66,
    0.93,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    animated=True,
)

mapping_label = ax.text(
    0.66,
    0.84,
    r"mapping",
    transform=ax.transAxes,
    ha="left",
    va="top",
    animated=True,
)

# Tiny schematic box in data coordinates
x0 = 1.47
y0 = 0.50 * ymax_bins
box_w = 0.34
box_h = 0.18 * ymax_bins

# miniature axes
(map_x_axis,) = ax.plot(
    [x0, x0 + box_w],
    [y0, y0],
    color="k",
    linewidth=1.5,
    zorder=45,
    animated=True,
)
(map_y_axis,) = ax.plot(
    [x0, x0],
    [y0, y0 + box_h],
    color="k",
    linewidth=1.5,
    zorder=45,
    animated=True,
)

# identity reference line
(map_ref_line,) = ax.plot(
    [x0, x0 + 0.82 * box_w],
    [y0, y0 + 0.82 * box_h],
    color="0.5",
    linewidth=1.3,
    linestyle="--",
    zorder=46,
    animated=True,
)

# animated mean-scale mapping line
(map_line,) = ax.plot(
    [],
    [],
    color="k",
    linewidth=2.3,
    zorder=50,
    animated=True,
)

map_x_label = ax.text(
    x0 + box_w,
    y0 - 0.035 * ymax_bins,
    r"$z$",
    ha="right",
    va="top",
    fontsize=15,
    animated=True,
)

map_y_label = ax.text(
    x0 - 0.035,
    y0 + box_h,
    r"$z_{\rm ph}$",
    ha="right",
    va="bottom",
    fontsize=15,
    rotation=90,
    animated=True,
)


def draw_frame(frame_idx):
    curves = all_curves[frame_idx]

    for b in range(n_bins):
        y = curves[b]
        fills_bins[b].set_xy(fill_vertices(z, y))
        lines_bins[b].set_data(z, y)

    alpha = mean_scale_vals[frame_idx]
    text_main.set_text(rf"$\alpha = {alpha:.3f}$")

    # Mapping schematic: steeper line for larger alpha
    # normalized slope relative to the box aspect
    ref_slope = box_h / box_w
    slope = alpha * ref_slope

    x_end_candidate = x0 + box_w
    y_end_candidate = y0 + slope * box_w

    if y_end_candidate <= y0 + box_h:
        x_end = x_end_candidate
        y_end = y_end_candidate
    else:
        y_end = y0 + box_h
        x_end = x0 + box_h / slope

    map_line.set_data([x0, x_end], [y0, y_end])

    return [
        *fills_bins,
        *lines_bins,
        text_main,
        mapping_label,
        map_x_axis,
        map_y_axis,
        map_ref_line,
        map_line,
        map_x_label,
        map_y_label,
    ]


def init():
    return draw_frame(frame_ids[0])


def update(i):
    return draw_frame(frame_ids[i])


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_ids),
    init_func=init,
    interval=1000 / FPS,
    blit=True,
    repeat=True,
)

anim.save(OUTFILE, writer=PillowWriter(fps=FPS))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
