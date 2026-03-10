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
            linewidth=2.2,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")
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

specz_spec = {
    "kind": "specz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 6,
        "range": (0.05, 0.8),
    },
    "normalize_bins": True,
}

specz_result = tomo.build_bins(z=z, nz=nz, tomo_spec=specz_spec)

fig, ax = plt.subplots(figsize=(8.2, 4.8))
plot_bins(
    ax,
    z,
    specz_result.bins,
    title="Spec-z binning: 6 equipopulated bins",
    xmax=0.5
)
plt.tight_layout()