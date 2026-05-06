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
tomo.build_bins(
    z=z,
    nz=nz,
    tomo_spec=photoz_spec,
    include_tomo_metadata=True,
)

stats = tomo.cross_bin_stats(
    overlap={"method": "min", "unit": "fraction", "normalize": True, "decimal_places": 6},
    pearson={"normalize": True, "decimal_places": 6},
)

overlap = stats["overlap"]
pearson = stats["pearson"]

keys = sorted(overlap.keys())

overlap_vals = []
pearson_vals = []
labels = []

for a, i in enumerate(keys):
    for j in keys[a + 1 :]:
        overlap_vals.append(overlap[i][j])
        pearson_vals.append(pearson[i][j])
        labels.append(f"({i+1}–{j+1})")

overlap_vals = np.array(overlap_vals)
pearson_vals = np.array(pearson_vals)

x = np.arange(len(labels))
width = 0.38

colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0.2, 0.8),
    return_fmt="hex",
)

c_overlap = to_rgba(colors[1], 0.6)
c_pearson = to_rgba(colors[3], 0.6)

fig, ax = plt.subplots(figsize=(7.4, 5.2))

ax.bar(
    x - width / 2,
    overlap_vals,
    width,
    color=c_overlap,
    edgecolor="k",
    linewidth=2.0,
    label="Min overlap",
)

ax.bar(
    x + width / 2,
    pearson_vals,
    width,
    color=c_pearson,
    edgecolor="k",
    linewidth=2.0,
    label="Pearson correlation",
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Bin pair")
ax.set_ylabel("Metric value")
ax.set_title("Overlap and Pearson correlation by bin pair")

ax.legend(frameon=True)

plt.tight_layout()