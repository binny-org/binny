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
            linewidth=1.8,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=1.8, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")

tomo = NZTomography()

z = np.linspace(0.0, 4.5, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.5,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

unified_uncertainty_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "scatter_scale": [0.03, 0.04, 0.05, 0.06],
        "mean_offset": [0.00, 0.01, 0.01, 0.02],
        "mean_scale": [1.00, 1.00, 1.00, 1.00],
        "outlier_frac": [0.00, 0.05, 0.1, 0.15],
        "outlier_scatter_scale": [0.00, 0.20, 0.25, 0.30],
        "outlier_mean_offset": [0.00, 0.05, 0.05, 0.08],
        "outlier_mean_scale": [1.00, 1.00, 1.00, 1.00],
    },
    "normalize_bins": True,
}

unified_result = tomo.build_bins(
    z=z,
    nz=nz,
    tomo_spec=unified_uncertainty_spec,
)

fig, ax = plt.subplots(figsize=(8.6, 4.9))
plot_bins(
    ax,
    z,
    unified_result.bins,
    title="Unified photo-z uncertainty model",
)
plt.tight_layout()