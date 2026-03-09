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
OUTFILE = OUTDIR / "specz_uncertainty_completeness.gif"

tomo = NZTomography()

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

completeness_vals = np.linspace(1.0, 0.2, 16)

# Build the baseline spec-z bins once
spec_fixed = {
    "kind": "specz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": n_bins,
        "range": bin_range,
    },
    "uncertainties": {
        "completeness": 1.0,
    },
    "normalize_bins": False,
}

result_fixed = tomo.build_bins(z=z, nz=nz, tomo_spec=spec_fixed)
fixed_curves = ordered_curves(result_fixed.bins)

# Make the completeness effect visually explicit:
# keep bins 1 and 2 fixed, scale only bin 3 by c3
all_curves = []
for c3 in completeness_vals:
    frame_curves = [
        fixed_curves[0].copy(),
        fixed_curves[1].copy(),
        fixed_curves[2].copy() * float(c3),
    ]
    all_curves.append(frame_curves)

frame_ids = make_pingpong_indices(len(completeness_vals))
ymax_bins = 1.08 * max(np.max(curve) for frame in all_curves for curve in frame)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

lw = 2.0

ax.set_title("Spec-z bins with decreasing completeness")
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
    0.62,
    0.93,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    animated=True,
)

proxy_label = ax.text(
    0.62,
    0.84,
    "completeness",
    transform=ax.transAxes,
    ha="left",
    va="top",
    animated=True,
)

# A vertical bar that shrinks as completeness decreases
(proxy_bar,) = ax.plot(
    [],
    [],
    color="k",
    linewidth=3.0,
    zorder=50,
    animated=True,
    transform=ax.transAxes,
)

# small top cap for the bar
(proxy_cap,) = ax.plot(
    [],
    [],
    color="k",
    linewidth=2.0,
    zorder=50,
    animated=True,
    transform=ax.transAxes,
)

bar_x = 0.74
bar_y0 = 0.63
bar_y1_max = 0.75
cap_halfwidth = 0.018


def draw_frame(frame_idx):
    curves = all_curves[frame_idx]
    artists = []

    for b in range(n_bins):
        y = curves[b]
        fills_bins[b].set_xy(fill_vertices(z, y))
        lines_bins[b].set_data(z, y)
        artists.append(fills_bins[b])
        artists.append(lines_bins[b])

    c3 = completeness_vals[frame_idx]
    text_main.set_text(rf"$c_3 = {c3:.2f}$")

    bar_y1 = bar_y0 + c3 * (bar_y1_max - bar_y0)
    proxy_bar.set_data([bar_x, bar_x], [bar_y0, bar_y1])
    proxy_cap.set_data(
        [bar_x - cap_halfwidth, bar_x + cap_halfwidth],
        [bar_y1, bar_y1],
    )

    artists.extend([text_main, proxy_label, proxy_bar, proxy_cap])
    return artists


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
