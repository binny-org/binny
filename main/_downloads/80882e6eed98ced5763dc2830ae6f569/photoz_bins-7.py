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
        ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
        ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")

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

no_outliers_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": {
        "scatter_scale": 0.04,
    },
    "normalize_bins": True,
}

with_outliers_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": {
        "scatter_scale": 0.04,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.25,
        "outlier_mean_offset": 0.06,
        "outlier_mean_scale": 1.50,
    },
    "normalize_bins": True,
}

no_outliers_result = tomo.build_bins(z=z, nz=nz, tomo_spec=no_outliers_spec)
with_outliers_result = tomo.build_bins(z=z, nz=nz, tomo_spec=with_outliers_spec)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

plot_bins(axes[0], z, no_outliers_result.bins, "outlier_frac = 0.00")
axes[0].set_ylabel(r"Normalized $n_i(z)$")

plot_bins(axes[1], z, with_outliers_result.bins, "outlier_frac = 0.08")

plt.tight_layout()