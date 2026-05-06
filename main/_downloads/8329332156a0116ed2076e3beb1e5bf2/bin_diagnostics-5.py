import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import to_rgba

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

photoz_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
    "uncertainties": {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.03,
        "outlier_scatter_scale": 0.20,
        "outlier_mean_offset": 0.05,
    },
    "normalize_bins": True,
}

tomo = NZTomography()
result = tomo.build_bins(
    z=z,
    nz=nz,
    tomo_spec=photoz_spec,
    include_tomo_metadata=True,
)

bin_edges = result.tomo_meta["bins"]["bin_edges"]

leakage = tomo.cross_bin_stats(
    leakage={"bin_edges": bin_edges, "unit": "percent", "decimal_places": 3},
)["leakage"]

keys = sorted(leakage.keys())

completeness = [leakage[key][key] for key in keys]
contamination = [100.0 - leakage[key][key] for key in keys]

x = np.arange(len(keys))
width = 0.38

colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0.15, 0.85),
    return_fmt="hex",
)

c_comp, c_cont = colors[1], colors[3]

fill_comp = to_rgba(c_comp, 0.6)
fill_cont = to_rgba(c_cont, 0.6)

fig, ax = plt.subplots(figsize=(7.8, 4.8))

ax.bar(
    x - width / 2,
    completeness,
    width=width,
    color=fill_comp,
    edgecolor="k",
    linewidth=2.5,
    label="Completeness",
)

ax.bar(
    x + width / 2,
    contamination,
    width=width,
    color=fill_cont,
    edgecolor="k",
    linewidth=2.5,
    label="Contamination",
)

ax.set_xticks(x)
ax.set_xticklabels([f"Bin {key + 1}" for key in keys])

ax.set_ylabel("[%]")
ax.set_xlabel("Tomographic bin")
ax.set_title("Leakage-based completeness and contamination")

ax.legend(frameon=False)

plt.tight_layout()