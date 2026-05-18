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

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")


results = {}

for role in ["lens", "source"]:
    tomo = NZTomography()
    results[role] = tomo.build_survey_bins(
        "des",
        role=role,
        year="y1",
        include_tomo_metadata=True,
    )


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.8),
)

panel_order = [
    ("lens", "DES lens bins"),
    ("source", "DES source bins"),
]

for ax, (role, title) in zip(axes, panel_order, strict=True):
    plot_bins(ax, results[role], title)

    if role == "lens":
        ax.set_xlim(0.0, 1.0)

axes[0].set_ylabel(r"Normalized $n_i(z)$")

plt.tight_layout()