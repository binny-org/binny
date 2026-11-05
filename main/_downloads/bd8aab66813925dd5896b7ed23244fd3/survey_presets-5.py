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


edges = {
    ("lrg", "one_bin"): [0.4, 1.0],
    ("lrg", "three_bins"): [0.4, 0.6, 0.8, 1.0],
    ("lrg", "five_bins"): [0.4, 0.52, 0.64, 0.76, 0.88, 1.0],
    ("elg", "one_bin"): [0.6, 1.5],
    ("elg", "three_bins"): [0.6, 0.9, 1.2, 1.5],
    ("elg", "five_bins"): [0.6, 0.78, 0.96, 1.14, 1.32, 1.5],
}


results = {}

for key, bin_edges in edges.items():
    sample, _ = key

    tomo = NZTomography()
    results[key] = tomo.build_survey_bins(
        "desi",
        role="lens",
        sample=sample,
        overrides={"bins": {"edges": bin_edges}},
        include_tomo_metadata=True,
    )


fig, axes = plt.subplots(
    3,
    2,
    figsize=(11.5, 11.0),
)

panel_order = [
    (("lrg", "one_bin"), "DESI LRG: one bin"),
    (("elg", "one_bin"), "DESI ELG: one bin"),
    (("lrg", "three_bins"), "DESI LRG: three bins"),
    (("elg", "three_bins"), "DESI ELG: three bins"),
    (("lrg", "five_bins"), "DESI LRG: five bins"),
    (("elg", "five_bins"), "DESI ELG: five bins"),
]

for ax, (key, title) in zip(axes.ravel(), panel_order, strict=True):
    plot_bins(ax, results[key], title)

axes[0, 0].set_xlim(0.35, 1.05)
axes[1, 0].set_xlim(0.35, 1.05)
axes[2, 0].set_xlim(0.35, 1.05)

axes[0, 1].set_xlim(0.55, 1.55)
axes[1, 1].set_xlim(0.55, 1.55)
axes[2, 1].set_xlim(0.55, 1.55)

axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")
axes[2, 0].set_ylabel(r"Normalized $n_i(z)$")

plt.suptitle("DESI survey preset tomography", fontsize=16)

plt.tight_layout(rect=(0, 0, 1, 0.97))