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


results = {}

for role in ["lens", "source"]:
    for year in ["1", "10"]:
        tomo = NZTomography()
        results[(role, year)] = tomo.build_survey_bins(
            "lsst",
            role=role,
            year=year,
            include_tomo_metadata=True,
        )


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 8.0),
)

panel_order = [
    (("lens", "1"), "Lens bins Y1"),
    (("source", "1"), "Source bins Y1"),
    (("lens", "10"), "Lens bins Y10"),
    (("source", "10"), "Source bins Y10"),
]

for ax, (key, title) in zip(axes.ravel(), panel_order, strict=True):
    plot_bins(ax, results[key], title)

    role, year = key
    if role == "lens":
        ax.set_xlim(0.0, 1.5)

axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")

plt.suptitle("LSST survey preset tomography", fontsize=16)

plt.tight_layout(rect=(0, 0, 1, 0.97))