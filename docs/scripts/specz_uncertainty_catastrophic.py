from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from binny import NZTomography
from binny.nz_tomo.specz import build_specz_response_matrix

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
HOLD_FRAMES = 4
FIGSIZE = (12.8, 6.2)


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


def make_spec_cfg(model, f2, n_bins, bin_range, leakage_sigma):
    spec_cfg = {
        "kind": "specz",
        "bins": {
            "scheme": "equidistant",
            "n_bins": n_bins,
            "range": bin_range,
        },
        "uncertainties": {
            "completeness": 1.0,
            # Focus on bin 2 only: Python index 1
            "catastrophic_frac": [0.0, float(f2), 0.0],
            "leakage_model": model,
        },
        "normalize_bins": False,
    }

    if model == "gaussian":
        spec_cfg["uncertainties"]["leakage_sigma"] = leakage_sigma

    return spec_cfg


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "specz_uncertainty_catastrophic.gif"

tomo = NZTomography()

z = np.linspace(0.0, 2.0, 220)

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

# Deliberately exaggerated for illustration
catastrophic_vals = np.linspace(0.30, 0.98, 18)

# Pick one model only
leakage_model = "neighbor"
panel_title = "Neighbor leakage"

leakage_sigma = 2.6

all_curves = []
all_response_matrices = []

for f2 in catastrophic_vals:
    spec_cfg = make_spec_cfg(
        model=leakage_model,
        f2=f2,
        n_bins=n_bins,
        bin_range=bin_range,
        leakage_sigma=leakage_sigma,
    )

    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec_cfg)
    curves = ordered_curves(result.bins)
    all_curves.append(curves)

    if leakage_model == "gaussian":
        matrix = build_specz_response_matrix(
            n_bins,
            catastrophic_frac=[0.0, float(f2), 0.0],
            leakage_model=leakage_model,
            leakage_sigma=leakage_sigma,
        )
    else:
        matrix = build_specz_response_matrix(
            n_bins,
            catastrophic_frac=[0.0, float(f2), 0.0],
            leakage_model=leakage_model,
        )

    all_response_matrices.append(np.asarray(matrix, dtype=float))

base_frame_ids = make_pingpong_indices(len(catastrophic_vals))
frame_ids = [idx for idx in base_frame_ids for _ in range(HOLD_FRAMES)]
ymax_bins = 1.08 * max(np.max(curve) for frame in all_curves for curve in frame)

fig, (ax_curve, ax_matrix) = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
)

ax_curve.set_box_aspect(1)
ax_matrix.set_box_aspect(1)

lw = 2.0
zero_vertices = fill_vertices(z, np.zeros_like(z))

# --------------------
# Left panel: bin curves
# --------------------
ax_curve.set_title(panel_title)
ax_curve.set_xlim(0.0, 2.0)
ax_curve.set_ylim(0.0, ymax_bins)
ax_curve.set_xlabel("Redshift $z$")
ax_curve.set_ylabel(r"$n_i(z)$")

for side in ["left", "right", "top", "bottom"]:
    ax_curve.spines[side].set_visible(True)
    ax_curve.spines[side].set_linewidth(lw)

ax_curve.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
    width=lw,
    length=6,
)

ax_curve.grid(False)
ax_curve.plot(z, np.zeros_like(z), color="k", linewidth=lw, zorder=100)

fills_bins = []
lines_bins = []

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
    ax_curve.add_patch(poly)
    fills_bins.append(poly)

    (line,) = ax_curve.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=lw,
        zorder=20,
        animated=True,
    )
    lines_bins.append(line)

text_main = ax_curve.text(
    0.68,
    0.93,
    "",
    transform=ax_curve.transAxes,
    ha="left",
    va="top",
    animated=True,
)

# --------------------
# Right panel: response matrix
# --------------------
matrix0 = all_response_matrices[0]

matrix_im = ax_matrix.imshow(
    matrix0,
    origin="lower",
    aspect="equal",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    alpha=0.6,
    interpolation="none",
    animated=True,
)

ax_matrix.set_title("Response matrix")
ax_matrix.set_xlabel(r"True bin $j$")
ax_matrix.set_ylabel(r"Observed bin $i$")

ax_matrix.set_xticks(np.arange(n_bins))
ax_matrix.set_yticks(np.arange(n_bins))
ax_matrix.set_xticklabels([f"{j + 1}" for j in range(n_bins)])
ax_matrix.set_yticklabels([f"{i + 1}" for i in range(n_bins)])

ax_matrix.set_xticks(np.arange(-0.5, n_bins, 1), minor=True)
ax_matrix.set_yticks(np.arange(-0.5, n_bins, 1), minor=True)
ax_matrix.grid(which="minor", color="k", linestyle="-", linewidth=2)
ax_matrix.tick_params(which="minor", bottom=False, left=False)

matrix_texts = []
for i in range(n_bins):
    row = []
    for j in range(n_bins):
        txt = ax_matrix.text(
            j,
            i,
            f"{matrix0[i, j]:.2f}",
            ha="center",
            va="center",
            color="k",
            animated=True,
        )
        row.append(txt)
    matrix_texts.append(row)


def draw_frame(frame_idx):
    artists = []
    f2 = catastrophic_vals[frame_idx]
    curves = all_curves[frame_idx]
    matrix = all_response_matrices[frame_idx]

    for b in range(n_bins):
        y = curves[b]
        fills_bins[b].set_xy(fill_vertices(z, y))
        lines_bins[b].set_data(z, y)
        artists.append(fills_bins[b])
        artists.append(lines_bins[b])

    text_main.set_text(rf"$f_2 = {f2:.2f}$")
    artists.append(text_main)

    matrix_im.set_data(matrix)
    artists.append(matrix_im)

    for i in range(n_bins):
        for j in range(n_bins):
            matrix_texts[i][j].set_text(f"{matrix[i, j]:.2f}")
            artists.append(matrix_texts[i][j])

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
    interval=180,
    blit=True,
    repeat=True,
)

anim.save(OUTFILE, writer=PillowWriter(fps=FPS))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
