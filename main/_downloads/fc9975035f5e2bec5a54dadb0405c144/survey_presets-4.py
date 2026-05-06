from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import yaml

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


def plot_bins(ax, z, bin_dict, title):
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


def plot_bins_dashed(ax, z, bin_dict):
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


def build_tomo_spec(entry):
    return {
        "kind": entry["kind"],
        "bins": entry["bins"],
        "uncertainties": entry["uncertainties"],
        "normalize_bins": True,
    }


# Load Roman preset
preset_path = Path("../../src/binny/surveys/configs/roman_survey_specs.yaml")

with preset_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# Redshift grid
z_cfg = config["z_grid"]

z = np.linspace(
    z_cfg["start"],
    z_cfg["stop"],
    z_cfg["n"],
)


# Select tomography entries
entries = config["tomography"]

selected = {
    ("lens", "hls_optimistic"): None,
    ("source", "hls_optimistic"): None,
    ("lens", "hls_conservative"): None,
    ("source", "hls_conservative"): None,
    ("lens", "wide"): None,
    ("source", "wide"): None,
}

for entry in entries:
    key = (entry["role"], entry["scenario"])
    if key in selected:
        selected[key] = entry


# Build tomography
results = {}

for key, entry in selected.items():

    nz = NZTomography.nz_model(
        entry["nz"]["model"],
        z,
        normalize=True,
        **entry["nz"]["params"],
    )

    tomo = NZTomography()

    result = tomo.build_bins(
        z=z,
        nz=nz,
        tomo_spec=build_tomo_spec(entry),
        include_tomo_metadata=True,
    )

    results[key] = result


# Plot layout:
# HLS lens     HLS source
# wide lens    wide source

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 8.0),
)

plot_bins(
    axes[0, 0],
    z,
    results[("lens", "hls_optimistic")].bins,
    "Roman HLS lens bins",
)
plot_bins_dashed(
    axes[0, 0],
    z,
    results[("lens", "hls_conservative")].bins,
)

plot_bins(
    axes[0, 1],
    z,
    results[("source", "hls_optimistic")].bins,
    "Roman HLS source bins",
)
plot_bins_dashed(
    axes[0, 1],
    z,
    results[("source", "hls_conservative")].bins,
)

plot_bins(
    axes[1, 0],
    z,
    results[("lens", "wide")].bins,
    "Roman wide lens bins",
)

plot_bins(
    axes[1, 1],
    z,
    results[("source", "wide")].bins,
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