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

common_uncertainties = {"scatter_scale": 0.05}

three_bins_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 3},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

five_bins_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 5},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

three_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=three_bins_spec)
five_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=five_bins_spec)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

plot_bins(axes[0], z, three_bin_result.bins, "3 bins")
axes[0].set_ylabel(r"Normalized $n_i(z)$")

plot_bins(axes[1], z, five_bin_result.bins, "5 bins")

plt.tight_layout()