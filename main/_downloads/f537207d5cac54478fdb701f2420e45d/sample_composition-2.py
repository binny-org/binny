import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography


def plot_bins(ax, result, title):
    z = result.z
    bin_dict = result.bins
    keys = sorted(bin_dict.keys())

    colors = cmr.take_cmap_colors(
        "viridis",
        len(keys),
        cmap_range=(0.1, 0.9),
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

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")


lrg_tomo = NZTomography()
elg_tomo = NZTomography()

lrg = lrg_tomo.build_survey_bins(
    "desi",
    role="lens",
    sample="lrg",
    overrides={"bins": {"edges": [0.4, 0.6, 0.8, 1.0]}},
)

elg = elg_tomo.build_survey_bins(
    "desi",
    role="lens",
    sample="elg",
    overrides={"bins": {"edges": [0.6, 0.9, 1.2, 1.5]}},
)

combined = NZTomography.combine_tomography_bins(
    [lrg, elg],
    interpolate=True,
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.6),
)

plot_bins(axes[0], lrg, "DESI LRG bins")
plot_bins(axes[1], elg, "DESI ELG bins")
plot_bins(axes[2], combined, "Combined LRG + ELG bins")

for ax in axes:
    ax.set_xlim(0.35, 1.55)

axes[0].set_ylabel(r"Normalized $n_i(z)$")

plt.suptitle("DESI sample-composed tomography", fontsize=16)

plt.tight_layout(rect=(0, 0, 1, 0.94))