import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography


def get_bin_colors(bin_dict):
    keys = sorted(bin_dict.keys())

    colors = cmr.take_cmap_colors(
        "viridis",
        len(keys),
        cmap_range=(0.1, 0.9),
        return_fmt="hex",
    )

    return keys, colors


def plot_bins(ax, result, title):
    z = result.z
    bin_dict = result.bins
    keys, colors = get_bin_colors(bin_dict)

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


def plot_bins_dashed(ax, result):
    z = result.z
    bin_dict = result.bins
    keys, colors = get_bin_colors(bin_dict)

    for i, key in enumerate(keys):
        curve = np.asarray(bin_dict[key], dtype=float)

        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            linestyle="--",
            zorder=120 + i,
        )


results = {}

for scenario in ["hls_optimistic", "hls_conservative", "wide"]:
    for role in ["lens", "source"]:
        tomo = NZTomography()
        results[(role, scenario)] = tomo.build_survey_bins(
            "roman",
            role=role,
            scenario=scenario,
            include_tomo_metadata=True,
        )


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 8.0),
)

plot_bins(
    axes[0, 0],
    results[("lens", "hls_optimistic")],
    "Roman HLS lens bins",
)
plot_bins_dashed(
    axes[0, 0],
    results[("lens", "hls_conservative")],
)

plot_bins(
    axes[0, 1],
    results[("source", "hls_optimistic")],
    "Roman HLS source bins",
)
plot_bins_dashed(
    axes[0, 1],
    results[("source", "hls_conservative")],
)

plot_bins(
    axes[1, 0],
    results[("lens", "wide")],
    "Roman wide lens bins",
)

plot_bins(
    axes[1, 1],
    results[("source", "wide")],
    "Roman wide source bins",
)

axes[0, 0].set_xlim(0.0, 4.0)
axes[0, 1].set_xlim(0.0, 4.0)
axes[1, 0].set_xlim(0.0, 4.0)
axes[1, 1].set_xlim(0.0, 4.0)

axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")

axes[0, 0].plot([], [], color="k", linewidth=1.8, label="HLS optimistic")
axes[0, 0].plot(
    [],
    [],
    color="k",
    linewidth=1.8,
    linestyle="--",
    label="HLS conservative",
)
axes[0, 0].legend(frameon=False)

plt.suptitle("Roman survey preset tomography", fontsize=16)

plt.tight_layout(rect=(0, 0, 1, 0.97))