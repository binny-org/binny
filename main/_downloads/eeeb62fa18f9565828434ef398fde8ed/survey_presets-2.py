from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import yaml

from binny import NZTomography


def plot_bins(ax, z, bin_dict, title):
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


def build_tomo_spec(entry):
    return {
        "kind": entry["kind"],
        "bins": entry["bins"],
        "uncertainties": entry["uncertainties"],
        "normalize_bins": entry.get("normalize_bins", True),
    }


# Load Euclid preset
preset_path = Path("../../src/binny/surveys/configs/euclid_survey_specs.yaml")

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
    ("lens", "nominal"): None,
    ("source", "nominal"): None,
}

for entry in entries:
    key = (entry["role"], entry["year"])
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


# Plot
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.8),
)

panel_order = [
    (("lens", "nominal"), "Euclid lens bins"),
    (("source", "nominal"), "Euclid source bins"),
]

for ax, (key, title) in zip(axes, panel_order, strict=True):
    plot_bins(ax, z, results[key].bins, title)

axes[0].set_xlim(0.75, 1.9)
axes[1].set_xlim(0.0, 2.6)

axes[0].set_ylabel(r"Normalized $n_i(z)$")

plt.tight_layout()