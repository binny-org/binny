"""Illustrate full-grid binning and alternative tomographic edge specifications."""

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.axes import equal_number_edges, equidistant_edges


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "binning_interval_modes.png"

# Parent redshift distribution.
z = np.linspace(0.0, 3.0, 500)
nz = z**2 * np.exp(-z / 0.5)
nz /= nz.max()

n_bins = 5

# Equidistant bins spanning the full input grid.
full_edges = equidistant_edges(z.min(), z.max(), n_bins)

# Explicit, nonuniform bin boundaries.
explicit_edges = np.array([0.25, 0.35, 0.65, 1.10, 1.63, 2.0])

# Equal-number bins over a restricted redshift interval.
equal_min, equal_max = 0.3, 2.4
z_equal = np.linspace(equal_min, equal_max, 500)
nz_equal = np.interp(z_equal, z, nz)
equal_edges = equal_number_edges(z_equal, nz_equal, n_bins)

# Mixed binning: two equidistant bins followed by three equal-number bins.
segment_specs = [
    {
        "scheme": "equidistant",
        "n_bins": 2,
        "x_min": 0.4,
        "x_max": 0.9,
    },
    {
        "scheme": "equal_number",
        "n_bins": 3,
        "x_min": 0.9,
        "x_max": 2.4,
    },
]

segment_edges = []

for segment in segment_specs:
    lo = segment["x_min"]
    hi = segment["x_max"]
    count = segment["n_bins"]

    if segment["scheme"] == "equidistant":
        edges = equidistant_edges(lo, hi, count)
    elif segment["scheme"] == "equal_number":
        z_segment = np.linspace(lo, hi, 500)
        nz_segment = np.interp(z_segment, z, nz)
        edges = equal_number_edges(z_segment, nz_segment, count)
    else:
        raise ValueError(f"Unknown scheme: {segment['scheme']}")

    segment_edges.append(edges)

mixed_edges = np.concatenate(
    [segment_edges[0], segment_edges[1][1:]]
)

# Use the same Viridis palette as the Binny tomography animations.
colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.facecolor": "white",
})

# One large panel above three smaller comparison panels.
fig = plt.figure(figsize=(8.5, 7), facecolor="white")

gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[3, 2],
    hspace=0.4,
    wspace=0.3,
    left=0.07,
    right=0.98,
    top=0.94,
    bottom=0.1,
)

ax_top = fig.add_subplot(gs[0, :])
axes = [fig.add_subplot(gs[1, i]) for i in range(3)]


def style_axis(ax, *, large=False):
    """Apply consistent styling to each redshift-distribution plot."""
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel(r"$n(z)$", fontsize=17 if large else 14)
    ax.set_xlabel(r"Redshift $z$", fontsize=15 if large else 13)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.5,
        length=6,
        labelsize=13 if large else 10,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    ax.grid(False)


def draw_bins(ax, edges):
    """Draw the parent distribution and its tomographic bin boundaries."""
    ax.plot(
        z,
        nz,
        color="k",
        linewidth=2.2,
        zorder=10,
    )

    for i, (lo, hi) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        mask = (z >= lo) & (z <= hi)
        x_bin = np.concatenate(([lo], z[mask], [hi]))
        y_bin = np.interp(x_bin, z, nz)

        ax.fill_between(
            x_bin,
            0.0,
            y_bin,
            color=colors[i],
            alpha=0.6,
            linewidth=0.0,
            zorder=2,
        )

    for edge in edges:
        ax.vlines(
            edge,
            0.0,
            np.interp(edge, z, nz),
            color="k",
            linewidth=1.1,
            zorder=5,
        )

    ax.scatter(
        edges,
        np.zeros_like(edges),
        color="k",
        s=30,
        zorder=15,
    )


# Full-grid equidistant binning.
style_axis(ax_top, large=True)
draw_bins(ax_top, full_edges)

ax_top.set_title(
    "Full grid equidistant binning",
    fontsize=19,
    pad=15,
)

# Alternative binning specifications.
examples = [
    ("Explicit edges", explicit_edges),
    ("Equipopulated binning", equal_edges),
    ("Mixed binning", mixed_edges),
]

for ax, (title, edges) in zip(axes, examples, strict=True):
    style_axis(ax)
    draw_bins(ax, edges)
    ax.set_title(title, fontsize=16, pad=12)

fig.savefig(
    OUTFILE,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15,
)

plt.close(fig)

print(f"Saved illustration to: {OUTFILE}")
print(f"Full-grid edges: {full_edges}")
print(f"Explicit edges: {explicit_edges}")
print(f"Equal-number edges: {equal_edges}")
print(f"Mixed edges: {mixed_edges}")
