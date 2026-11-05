import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from binny import NZTomography


def plot_one_sample(
    ax,
    z,
    bins,
    title,
    cmap="viridis",
    cmap_range=(0.10, 0.90),
):
    keys = sorted(bins.keys())
    colors = cmr.take_cmap_colors(
        cmap,
        len(keys),
        cmap_range=cmap_range,
        return_fmt="hex",
    )

    for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
        curve = np.asarray(bins[key], dtype=float)
        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=0.65,
            linewidth=0.0,
            zorder=10 + i,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")


def square_triangles(x, y, size=0.42):
    x0, x1 = x - size, x + size
    y0, y1 = y - size, y + size

    tri1 = [(x0, y0), (x1, y0), (x0, y1)]
    tri2 = [(x1, y1), (x1, y0), (x0, y1)]
    border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    diag = [(x1, y0), (x0, y1)]

    return tri1, tri2, border, diag


def draw_pair_cell(ax, row, col, color_i, color_j, size=0.42):
    tri1, tri2, border, diag = square_triangles(col, row, size=size)

    ax.add_patch(
        Polygon(
            tri1,
            closed=True,
            facecolor=color_i,
            edgecolor="none",
            alpha=0.65,
            zorder=3,
        )
    )

    ax.add_patch(
        Polygon(
            tri2,
            closed=True,
            facecolor=color_j,
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


def setup_square_pair_axes(ax, n_bins, title):
    ax.set_title(title)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(n_bins - 0.5, -0.5)

    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels([f"{j + 1}" for j in range(n_bins)])
    ax.set_yticklabels([f"{i + 1}" for i in range(n_bins)])

    ax.set_xlabel("Lens bin $j$")
    ax.set_ylabel("Lens bin $i$")

    for k in range(n_bins + 1):
        ax.axhline(k - 0.5, color="k", linewidth=1.0, zorder=1)
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


def nested_dict_to_matrix(nested_dict):
    keys = sorted(nested_dict.keys())
    matrix = np.array(
        [[nested_dict[row_key][col_key] for col_key in keys] for row_key in keys],
        dtype=float,
    )
    return keys, matrix


z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
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
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.03,
        "outlier_scatter_scale": 0.20,
        "outlier_mean_offset": 0.05,
    },
    "normalize_bins": True,
}

lens = NZTomography()
lens_result = lens.build_bins(
    z=z,
    nz=nz,
    tomo_spec=lens_spec,
    include_tomo_metadata=True,
)

bin_edges = lens_result.tomo_meta["bins"]["bin_edges"]

leakage = lens.cross_bin_stats(
    leakage={"bin_edges": bin_edges, "unit": "fraction", "decimal_places": 6},
)["leakage"]

lens_keys, leakage_matrix = nested_dict_to_matrix(leakage)
n_lens = len(lens_keys)

# Build a symmetric leakage diagnostic by taking the larger
# of the two directional leakage values for each pair.
sym_leakage_matrix = np.maximum(leakage_matrix, leakage_matrix.T)

threshold = 0.12

candidate_pairs = [
    (i_key, j_key)
    for a, i_key in enumerate(lens_keys)
    for j_key in lens_keys[a:]
]

retained_pairs = []
excluded_pairs = []

for i_key, j_key in candidate_pairs:
    i = lens_keys.index(i_key)
    j = lens_keys.index(j_key)

    if i == j:
        retained_pairs.append((i_key, j_key))
    elif sym_leakage_matrix[i, j] <= threshold:
        retained_pairs.append((i_key, j_key))
    else:
        excluded_pairs.append((i_key, j_key))

lens_pos = {key: idx for idx, key in enumerate(lens_keys)}

lens_colors = cmr.take_cmap_colors(
    "viridis",
    n_lens,
    cmap_range=(0.10, 0.90),
    return_fmt="hex",
)

fig = plt.figure(figsize=(15.2, 8.8))
gs = fig.add_gridspec(
    2, 3, height_ratios=[1.0, 1.0],
    hspace=0.38, wspace=0.28)

ax_top = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])

# Top panel: lens bins in redshift space
plot_one_sample(
    ax_top,
    z,
    lens_result.bins,
    "Lens tomographic bins",
)

# Panel 1: candidate symmetric topology
setup_square_pair_axes(ax0, n_lens, "Candidate pairs")
for i_key, j_key in candidate_pairs:
    i = lens_pos[i_key]
    j = lens_pos[j_key]
    draw_pair_cell(ax0, i, j, lens_colors[i], lens_colors[j])

# Mask lower triangle so the unique-pair topology is visually clear
for i in range(n_lens):
    for j in range(i):
        ax0.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="white",
                edgecolor="none",
                zorder=20,
            )
        )

# Re-draw grid lines on top of the mask
for k in range(n_lens + 1):
    ax0.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
    ax0.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

# Panel 2: symmetrized leakage matrix
ax1.imshow(
    sym_leakage_matrix,
    origin="upper",
    aspect="auto",
    cmap="viridis",
    alpha=0.65,
    interpolation="none",
)

ax1.set_title("Leakage matrix")
ax1.set_xticks(np.arange(n_lens))
ax1.set_yticks(np.arange(n_lens))
ax1.set_xticklabels([f"{k + 1}" for k in lens_keys])
ax1.set_yticklabels([f"{k + 1}" for k in lens_keys])
ax1.set_xlabel("Lens bin $j$")
ax1.set_ylabel("Lens bin $i$")

ax1.set_xticks(np.arange(-0.5, n_lens, 1), minor=True)
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
    for j in range(n_lens):
        value = sym_leakage_matrix[i, j]
        txt = f"{value:.2f}"
        color = "k" if value > threshold else "white"
        ax1.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            color=color,
            zorder=5,
        )

# Panel 3: retained pairs after leakage cut
setup_square_pair_axes(
    ax2,
    n_lens,
    ax2_title := rf"Retained pairs $\leq {100 * threshold:.0f}\%$",
)

for i_key, j_key in candidate_pairs:
    i = lens_pos[i_key]
    j = lens_pos[j_key]
    draw_pair_cell(ax2, i, j, lens_colors[i], lens_colors[j])

for i in range(n_lens):
    for j in range(i):
        ax2.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="white",
                edgecolor="none",
                zorder=20,
            )
        )

for i_key, j_key in excluded_pairs:
    i = lens_pos[i_key]
    j = lens_pos[j_key]
    draw_exclusion_overlay(ax2, i, j)

for k in range(n_lens + 1):
    ax2.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
    ax2.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

plt.tight_layout()