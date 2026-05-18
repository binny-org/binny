import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

common_uncertainties = {
    "scatter_scale": [0.010, 0.012, 0.015, 0.018],
    "mean_offset": 0.0,
    "outlier_frac": [0.02, 0.05, 0.15, 0.26],
    "outlier_scatter_scale": [0.008, 0.010, 0.012, 0.015],
    "outlier_mean_offset": [0.35, 0.40, 0.45, 0.50],
}

equipopulated_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equidistant", "n_bins": 4, "range": (0.2, 1.2)},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

tomo_eqpop = NZTomography()
tomo_eqpop.build_bins(z=z, nz=nz, tomo_spec=equipopulated_spec)

tomo_eqdist = NZTomography()
tomo_eqdist.build_bins(z=z, nz=nz, tomo_spec=equidistant_spec)

shape_eqpop = tomo_eqpop.shape_stats(center_method="median")
shape_eqdist = tomo_eqdist.shape_stats(center_method="median")

keys = sorted(shape_eqpop["per_bin"].keys())
x = np.arange(len(keys))
width = 0.36

widths_eqpop = [
    shape_eqpop["per_bin"][k]["moments"]["width_68"] for k in keys
]
widths_eqdist = [
    shape_eqdist["per_bin"][k]["moments"]["width_68"] for k in keys
]

center_methods = ["mean", "median", "mode"]

centers_eqpop = {
    m: tomo_eqpop.shape_stats(center_method=m)["centers"] for m in center_methods
}

centers_eqdist = {
    m: tomo_eqdist.shape_stats(center_method=m)["centers"] for m in center_methods
}

colors = cmr.take_cmap_colors(
    "viridis",
    2,
    cmap_range=(0.2, 0.8),
    return_fmt="hex",
)

c_eqpop, c_eqdist = colors
fill_eqpop = to_rgba(c_eqpop, 0.6)
fill_eqdist = to_rgba(c_eqdist, 0.6)

marker_map = {
    "mean": "o",
    "median": "s",
    "mode": "^",
}

offset_map = {
    "mean": -0.18,
    "median": 0.0,
    "mode": 0.18,
}

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))

axes[0].bar(
    x - width / 2,
    widths_eqpop,
    width=width,
    color=fill_eqpop,
    edgecolor="k",
    linewidth=2,
    label="Equipopulated",
)

axes[0].bar(
    x + width / 2,
    widths_eqdist,
    width=width,
    color=fill_eqdist,
    edgecolor="k",
    linewidth=2,
    label="Equidistant",
)

axes[0].set_title("Central 68% widths")
axes[0].set_xlabel("Tomographic bin")
axes[0].set_ylabel("Width in redshift")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{k+1}" for k in keys])
axes[0].legend(frameon=False)

for m in center_methods:
    axes[1].scatter(
        x + offset_map[m] - 0.02,
        [centers_eqpop[m][k] for k in keys],
        marker=marker_map[m],
        s=160,
        color=fill_eqpop,
        edgecolor="k",
        linewidth=1.5,
        zorder=3,
    )

    axes[1].scatter(
        x + offset_map[m] + 0.02,
        [centers_eqdist[m][k] for k in keys],
        marker=marker_map[m],
        s=160,
        color=fill_eqdist,
        edgecolor="k",
        linewidth=1.5,
        zorder=3,
    )

scheme_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        linewidth=2,
        markerfacecolor=fill_eqpop,
        markeredgecolor="k",
        markersize=10,
        label="Equipopulated",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        linewidth=2,
        markerfacecolor=fill_eqdist,
        markeredgecolor="k",
        markersize=10,
        label="Equidistant",
    ),
]

shape_handles = [
    Line2D(
        [0], [0],
        marker=marker_map["mean"],
        linestyle="None",
        linewidth=2,
        color="k",
        markerfacecolor="None",
        markersize=10,
        label="Mean",
    ),
    Line2D(
        [0], [0],
        marker=marker_map["median"],
        linestyle="None",
        linewidth=2,
        color="k",
        markerfacecolor="None",
        markersize=10,
        label="Median",
    ),
    Line2D(
        [0], [0],
        marker=marker_map["mode"],
        linestyle="None",
        linewidth=2,
        color="k",
        markerfacecolor="None",
        markersize=10,
        label="Mode",
    ),
]

legend1 = axes[1].legend(
    handles=scheme_handles,
    loc="upper left",
    frameon=True,
)
axes[1].add_artist(legend1)

axes[1].legend(
    handles=shape_handles,
    loc="lower right",
    frameon=True,
)

axes[1].set_title("Peak-location summaries")
axes[1].set_xlabel("Tomographic bin")
axes[1].set_ylabel("Representative redshift")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{k+1}" for k in keys])

plt.tight_layout()