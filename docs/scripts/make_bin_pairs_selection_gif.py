from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle

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

# Animation controls
FPS = 8
PAUSE_FRAMES = 4
FINAL_PAUSE_FRAMES = 12
FIGSIZE = (15.2, 8.8)

# Thresholds to animate through
THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20]


def plot_two_samples(
    ax,
    z,
    lens_bins,
    source_bins,
    title,
    lens_cmap="viridis",
    source_cmap="viridis",
    lens_cmap_range=(0.10, 0.80),
    source_cmap_range=(0.20, 1.00),
):
    ax.cla()

    lens_keys = sorted(lens_bins.keys())
    source_keys = sorted(source_bins.keys())

    lens_colors = cmr.take_cmap_colors(
        lens_cmap,
        len(lens_keys),
        cmap_range=lens_cmap_range,
        return_fmt="hex",
    )
    source_colors = cmr.take_cmap_colors(
        source_cmap,
        len(source_keys),
        cmap_range=source_cmap_range,
        return_fmt="hex",
    )

    # Lenses: hatched + dashed
    for i, (color, key) in enumerate(zip(lens_colors, lens_keys, strict=True)):
        curve = np.asarray(lens_bins[key], dtype=float)
        ax.fill_between(
            z,
            0.0,
            curve,
            facecolor=color,
            alpha=0.65,
            linewidth=0.0,
            hatch="///",
            edgecolor=color,
            zorder=10 + i,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            linestyle="--",
            zorder=20 + i,
        )

    # Sources: solid filled
    for i, (color, key) in enumerate(zip(source_colors, source_keys, strict=True)):
        curve = np.asarray(source_bins[key], dtype=float)
        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=0.65,
            linewidth=0.0,
            zorder=40 + i,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            zorder=50 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")

    ax.text(
        0.98,
        0.96,
        "Hatched dashed: lens bins\nSolid filled: source bins",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="none",
        ),
    )


def square_triangles(x, y, size=0.42):
    x0, x1 = x - size, x + size
    y0, y1 = y - size, y + size

    tri1 = [(x0, y0), (x1, y0), (x0, y1)]
    tri2 = [(x1, y1), (x1, y0), (x0, y1)]
    border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    diag = [(x1, y0), (x0, y1)]

    return tri1, tri2, border, diag


def draw_pair_cell(ax, row, col, lens_color, source_color, size=0.42):
    tri1, tri2, border, diag = square_triangles(col, row, size=size)

    # Lower-left triangle: lens -> hatched
    ax.add_patch(
        Polygon(
            tri1,
            closed=True,
            facecolor=lens_color,
            edgecolor=lens_color,
            hatch="///",
            linewidth=0.0,
            alpha=0.65,
            zorder=3,
        )
    )

    # Upper-right triangle: source -> solid filled
    ax.add_patch(
        Polygon(
            tri2,
            closed=True,
            facecolor=source_color,
            edgecolor="none",
            alpha=0.65,
            zorder=3,
        )
    )

    ax.add_patch(
        Polygon(
            border,
            closed=True,
            facecolor="none",
            edgecolor="k",
            linewidth=1.8,
            zorder=4,
        )
    )
    ax.plot(
        [diag[0][0], diag[1][0]],
        [diag[0][1], diag[1][1]],
        color="k",
        linewidth=1.2,
        zorder=5,
    )


def draw_exclusion_overlay(ax, row, col, size=0.42):
    x0 = col - size
    y0 = row - size
    width = 2.0 * size
    height = 2.0 * size

    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="0.85",
            edgecolor="k",
            linewidth=1.5,
            alpha=0.65,
            zorder=10,
        )
    )
    ax.plot(
        [x0, x0 + width],
        [y0, y0 + height],
        color="k",
        linewidth=2.0,
        zorder=11,
    )
    ax.plot(
        [x0, x0 + width],
        [y0 + height, y0],
        color="k",
        linewidth=2.0,
        zorder=11,
    )


def setup_rect_pair_axes(ax, n_rows, n_cols, title):
    ax.cla()

    ax.set_title(title)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([f"{j + 1}" for j in range(n_cols)])
    ax.set_yticklabels([f"{i + 1}" for i in range(n_rows)])

    ax.set_xlabel("Source bin $j$")
    ax.set_ylabel("Lens bin $i$")

    for k in range(n_rows + 1):
        ax.axhline(k - 0.5, color="k", linewidth=1.0, zorder=1)
    for k in range(n_cols + 1):
        ax.axvline(k - 0.5, color="k", linewidth=1.0, zorder=1)

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.8)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.5,
        length=5,
    )

    ax.grid(False)


def nested_rect_dict_to_matrix(nested_dict):
    row_keys = sorted(nested_dict.keys())
    col_keys = sorted(nested_dict[row_keys[0]].keys())
    matrix = np.array(
        [[nested_dict[row_key][col_key] for col_key in col_keys] for row_key in row_keys],
        dtype=float,
    )
    return row_keys, col_keys, matrix


def get_selected_pairs(lens, source, threshold):
    spec = {
        "topology": {"name": "pairs_cartesian"},
        "filters": [
            {
                "name": "overlap_fraction",
                "threshold": threshold,
                "compare": "le",
            }
        ],
    }
    selected_pairs_raw = lens.bin_combo_filter(spec, other=source)
    return set(tuple(pair) for pair in selected_pairs_raw)


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "lens_source_pair_exclusions.gif"

# Build lens and source samples
z = np.linspace(0.0, 2.5, 600)

lens_nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.18,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

source_nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.32,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

lens_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
    "uncertainties": {
        "scatter_scale": 0.03,
        "mean_offset": 0.00,
        "outlier_frac": 0.01,
        "outlier_scatter_scale": 0.10,
        "outlier_mean_offset": 0.03,
    },
    "normalize_bins": True,
}

source_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "scatter_scale": 0.06,
        "mean_offset": 0.01,
        "outlier_frac": 0.04,
        "outlier_scatter_scale": 0.25,
        "outlier_mean_offset": 0.06,
    },
    "normalize_bins": True,
}

lens = NZTomography()
lens_result = lens.build_bins(
    z=z,
    nz=lens_nz,
    tomo_spec=lens_spec,
    include_tomo_metadata=True,
)

source = NZTomography()
source_result = source.build_bins(
    z=z,
    nz=source_nz,
    tomo_spec=source_spec,
    include_tomo_metadata=True,
)

between_overlap = lens.between_sample_stats(
    source,
    overlap={"method": "min", "unit": "fraction", "normalize": True, "decimal_places": 6},
)["overlap"]

lens_keys, source_keys, overlap_matrix = nested_rect_dict_to_matrix(between_overlap)

n_lens = len(lens_keys)
n_source = len(source_keys)

candidate_pairs = [(i_key, j_key) for i_key in lens_keys for j_key in source_keys]

lens_pos = {key: idx for idx, key in enumerate(lens_keys)}
source_pos = {key: idx for idx, key in enumerate(source_keys)}

lens_colors = cmr.take_cmap_colors(
    "viridis",
    n_lens,
    cmap_range=(0.10, 0.80),
    return_fmt="hex",
)

source_colors = cmr.take_cmap_colors(
    "viridis",
    n_source,
    cmap_range=(0.20, 1.00),
    return_fmt="hex",
)

# Precompute exclusions for each threshold
states = []
for threshold in THRESHOLDS:
    selected_pairs = get_selected_pairs(lens, source, threshold)
    excluded_pairs = [pair for pair in candidate_pairs if pair not in selected_pairs]
    states.append(
        {
            "threshold": threshold,
            "selected_pairs": selected_pairs,
            "excluded_pairs": excluded_pairs,
        }
    )

# Build animation timeline
timeline = []
for idx in range(len(states) - 1):
    timeline.extend([idx] * PAUSE_FRAMES)
timeline.extend([len(states) - 1] * FINAL_PAUSE_FRAMES)

# Figure layout
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.0, 1.0],
    hspace=0.38,
    wspace=0.28,
)

ax_top = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])

fig.suptitle("Lens-source pair selection across overlap thresholds")


def draw_static_panels(current_threshold):
    # Top panel: lens and source bins in redshift space
    plot_two_samples(
        ax_top,
        z,
        lens_result.bins,
        source_result.bins,
        "Lens and source tomographic bins",
    )

    # Panel 1: candidate Cartesian topology
    setup_rect_pair_axes(ax0, n_lens, n_source, "Candidate bin pairs")
    for i_key, j_key in candidate_pairs:
        i = lens_pos[i_key]
        j = source_pos[j_key]
        draw_pair_cell(ax0, i, j, lens_colors[i], source_colors[j])

    # Panel 2: between-sample overlap matrix
    ax1.cla()
    ax1.imshow(
        overlap_matrix,
        origin="upper",
        aspect="auto",
        cmap="viridis",
        alpha=0.65,
        interpolation="none",
        vmin=0.0,
        vmax=np.max(overlap_matrix),
    )

    ax1.set_title(
        rf"Overlap matrix ($\tau = {100 * current_threshold:.0f}\%$)"
    )
    ax1.set_xticks(np.arange(n_source))
    ax1.set_yticks(np.arange(n_lens))
    ax1.set_xticklabels([f"{k + 1}" for k in source_keys])
    ax1.set_yticklabels([f"{k + 1}" for k in lens_keys])
    ax1.set_xlabel("Source bin $j$")
    ax1.set_ylabel("Lens bin $i$")

    ax1.set_xticks(np.arange(-0.5, n_source, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n_lens, 1), minor=True)
    ax1.grid(which="minor", color="k", linestyle="-", linewidth=1.2)
    ax1.tick_params(which="minor", bottom=False, left=False)

    for side in ["left", "right", "top", "bottom"]:
        ax1.spines[side].set_visible(True)
        ax1.spines[side].set_linewidth(1.8)

    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.5,
        length=5,
    )

    for i in range(n_lens):
        for j in range(n_source):
            value = overlap_matrix[i, j]
            txt = f"{value:.2f}"
            color = "k" if value > current_threshold else "white"
            ax1.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=color,
                zorder=5,
            )


def update(frame):
    state = states[timeline[frame]]
    threshold = state["threshold"]
    excluded_pairs = state["excluded_pairs"]

    draw_static_panels(threshold)

    # Panel 3: excluded pairs for current threshold
    setup_rect_pair_axes(
        ax2,
        n_lens,
        n_source,
        rf"Excluded pairs ($>{100 * threshold:.0f}\%$, $\tau$)"
    )

    for i_key, j_key in candidate_pairs:
        i = lens_pos[i_key]
        j = source_pos[j_key]
        draw_pair_cell(ax2, i, j, lens_colors[i], source_colors[j])

    for i_key, j_key in excluded_pairs:
        i = lens_pos[i_key]
        j = source_pos[j_key]
        draw_exclusion_overlay(ax2, i, j)

    return []


anim = FuncAnimation(
    fig,
    update,
    frames=len(timeline),
    interval=1000 / FPS,
    blit=False,
    repeat=True,
)

anim.save(
    OUTFILE,
    writer=PillowWriter(fps=FPS),
)

plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
