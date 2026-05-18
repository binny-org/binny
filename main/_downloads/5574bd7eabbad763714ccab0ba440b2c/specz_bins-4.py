import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

def plot_bins(ax, z, bin_dict, title, xmin=0, xmax=1):
    keys = sorted(bin_dict.keys())
    colors = cmr.take_cmap_colors(
        "viridis",
        len(keys),
        cmap_range=(0.0, 1.0),
        return_fmt="hex",
    )

    for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
        curve = np.asarray(bin_dict[key], dtype=float)
        ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
        ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_xlim(xmin, xmax)

tomo = NZTomography()

z = np.linspace(0.0, 1.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.12,
    alpha=2.0,
    beta=1.5,
    normalize=True,
)

low_scatter_spec = {
    "kind": "specz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "completeness": [1.0, 1.0, 1.0, 1.0],
        "specz_scatter": [0.0005, 0.0005, 0.00075, 0.001],
    },
    "normalize_bins": True,
}

high_scatter_spec = {
    "kind": "specz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "completeness": [1.0, 1.0, 1.0, 1.0],
        "specz_scatter": [0.003, 0.003, 0.004, 0.005],
    },
    "normalize_bins": True,
}

low_scatter = tomo.build_bins(z=z, nz=nz, tomo_spec=low_scatter_spec)
high_scatter = tomo.build_bins(
    z=z,
    nz=nz,
    tomo_spec=high_scatter_spec,
)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

plot_bins(axes[0], z, low_scatter.bins, "Low spec-z scatter", xmax=0.5)
axes[0].set_ylabel(r"Normalized $n_i(z)$")

plot_bins(axes[1], z, high_scatter.bins, "High spec-z scatter", xmax=0.5)

plt.tight_layout()