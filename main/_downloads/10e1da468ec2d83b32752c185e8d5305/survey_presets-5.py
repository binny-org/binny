from copy import deepcopy
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
        "normalize_bins": True,
    }


def load_tabulated_nz(entry, z, data_dir):
    source = entry["nz"]["source"]
    path = data_dir / source["path"]

    table = np.loadtxt(
        path,
        skiprows=source.get("skiprows", 0),
    )

    z_file = table[:, source.get("z_col", 0)]
    nz_file = table[:, source.get("nz_col", 1)]

    nz = np.interp(
        z,
        z_file,
        nz_file,
        left=0.0,
        right=0.0,
    )

    if entry["nz"].get("params", {}).get("normalize", False):
        norm = np.trapezoid(nz, z)
        if norm > 0.0:
            nz = nz / norm

    return nz


def build_result(entry, z, data_dir, edges):
    local_entry = deepcopy(entry)
    local_entry["bins"]["edges"] = edges

    nz = load_tabulated_nz(local_entry, z, data_dir)

    tomo = NZTomography()

    return tomo.build_bins(
        z=z,
        nz=nz,
        tomo_spec=build_tomo_spec(local_entry),
        include_tomo_metadata=True,
    )


# Load DESI preset
preset_path = Path("../../src/binny/surveys/configs/desi_survey_specs.yaml")
data_dir = Path("../../src/binny/surveys/data")

with preset_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# Redshift grid
z_cfg = config["z_grid"]

z = np.linspace(
    z_cfg["start"],
    z_cfg["stop"],
    z_cfg["n"],
)


# Select DESI entries
entries = config["tomography"]

selected = {
    "lrg": None,
    "elg": None,
}

for entry in entries:
    if entry["role"] == "lens" and entry["year"] in selected:
        selected[entry["year"]] = entry


# One-bin, three-bin, and five-bin DESI windows
edges = {
    ("lrg", "one_bin"): [0.4, 1.0],
    ("lrg", "three_bins"): [0.4, 0.6, 0.8, 1.0],
    ("lrg", "five_bins"): [0.4, 0.52, 0.64, 0.76, 0.88, 1.0],
    ("elg", "one_bin"): [0.6, 1.5],
    ("elg", "three_bins"): [0.6, 0.9, 1.2, 1.5],
    ("elg", "five_bins"): [0.6, 0.78, 0.96, 1.14, 1.32, 1.5],
}


# Build tomography
results = {}

for key, bin_edges in edges.items():
    tracer, _ = key

    results[key] = build_result(
        selected[tracer],
        z,
        data_dir,
        bin_edges,
    )


# Plot layout:
# LRG one bin       ELG one bin
# LRG three bins    ELG three bins
# LRG five bins     ELG five bins

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
    plot_bins(ax, z, results[key].bins, title)

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