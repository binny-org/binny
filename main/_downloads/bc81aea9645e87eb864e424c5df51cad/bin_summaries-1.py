import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
    keys = sorted(bin_dict.keys())
    colors = cmr.take_cmap_colors(
        cmap,
        len(keys),
        cmap_range=cmap_range,
        return_fmt="hex",
    )

    for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
        curve = np.asarray(bin_dict[key], dtype=float)
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
            linewidth=2.0,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")

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
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equipopulated_result = NZTomography().build_bins(
    z=z,
    nz=nz,
    tomo_spec=equipopulated_spec,
    include_tomo_metadata=True,
)

equidistant_result = NZTomography().build_bins(
    z=z,
    nz=nz,
    tomo_spec=equidistant_spec,
    include_tomo_metadata=True,
)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

plot_bins(axes[0], z, equipopulated_result.bins, "Equipopulated bins")
axes[0].set_ylabel(r"Normalized $n_i(z)$")

plot_bins(axes[1], z, equidistant_result.bins, "Equidistant bins")

plt.tight_layout()