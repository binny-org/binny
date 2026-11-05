import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

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

tomo_equipopulated = NZTomography()
tomo_equipopulated.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equipopulated_spec,
    include_tomo_metadata=True,
)

tomo_equidistant = NZTomography()
tomo_equidistant.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equidistant_spec,
    include_tomo_metadata=True,
)

center_methods = ["mean", "median", "mode"]

centers_equipopulated = {
    method: tomo_equipopulated.shape_stats(
        center_method=method,
        decimal_places=3,
    )["centers"]
    for method in center_methods
}

centers_equidistant = {
    method: tomo_equidistant.shape_stats(
        center_method=method,
        decimal_places=3,
    )["centers"]
    for method in center_methods
}

keys = sorted(centers_equipopulated["mean"].keys())
x = np.arange(len(keys))

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

fill_colors = {
    m: to_rgba(c, 0.6)
    for m, c in zip(center_methods, colors)
}

marker_map = {
    "mean": "o",
    "median": "s",
    "mode": "^",
}

label_map = {
    "mean": "Mean",
    "median": "Median",
    "mode": "Mode",
}

offset_map = {
    "mean": -0.18,
    "median": 0.0,
    "mode": 0.18,
}

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)

scheme_data = [
    (axes[0], "Equipopulated bins", centers_equipopulated),
    (axes[1], "Equidistant bins", centers_equidistant),
]

for ax, title, centers in scheme_data:

    for method in center_methods:
        ax.scatter(
            x + offset_map[method],
            [centers[method][key] for key in keys],
            marker=marker_map[method],
            s=200,
            facecolors=fill_colors[method],
            edgecolors="k",
            linewidths=1.6,
            label=label_map[method],
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{key+1}" for key in keys])
    ax.set_xlabel("Tomographic bin")
    ax.legend(frameon=True)

axes[0].set_ylabel("Representative redshift")

plt.tight_layout()