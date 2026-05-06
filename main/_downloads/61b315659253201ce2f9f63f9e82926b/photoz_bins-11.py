import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

def plot_bins(
    ax,
    z,
    bin_dict,
    title,
    scatter_text,
    cmap="viridis",
    cmap_range=(0.0, 1.0),
):
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
            linewidth=2.2,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")

    ax.text(
        0.97,
        0.95,
        scatter_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="none",
        ),
    )

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

shared_uncertainty_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": {
        "scatter_scale": 0.05,
    },
    "normalize_bins": True,
}

per_bin_scatter = [0.08, 0.15, 0.22, 0.35]

per_bin_uncertainty_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": {
        "scatter_scale": per_bin_scatter,
    },
    "normalize_bins": True,
}

shared_result = tomo.build_bins(z=z, nz=nz, tomo_spec=shared_uncertainty_spec)
per_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=per_bin_uncertainty_spec)

shared_text = r"$\sigma_z = {:.2f}$".format(
    shared_uncertainty_spec["uncertainties"]["scatter_scale"]
)

per_bin_text = "\n".join(
    rf"$\sigma_{{z,{i+1}}} = {val:.2f}$"
    for i, val in enumerate(per_bin_scatter)
)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

plot_bins(
    axes[0],
    z,
    shared_result.bins,
    "Shared scatter",
    shared_text,
)
axes[0].set_ylabel(r"Normalized $n_i(z)$")

plot_bins(
    axes[1],
    z,
    per_bin_result.bins,
    "Per-bin scatter",
    per_bin_text,
)

plt.tight_layout()