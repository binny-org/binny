import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
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

pair_list = tomo.cross_bin_stats(
    pairs={
        "method": "min",
        "unit": "percent",
        "threshold": 0.0,
        "direction": "high",
        "normalize": True,
        "decimal_places": 3,
    }
)["correlations"]

labels = [f"({i+1} – {j+1})" for i, j, _ in pair_list]
values = np.array([value for _, _, value in pair_list])
y = np.arange(len(labels))

colors = cmr.take_cmap_colors(
    "viridis",
    len(labels),
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)
fill_colors = [to_rgba(color, 0.6) for color in colors]

fig, ax = plt.subplots(figsize=(7.8, 4.8))

ax.barh(
    y,
    values,
    color=fill_colors,
    edgecolor="k",
    linewidth=2.5,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Min overlap [%]")
ax.set_ylabel("Bin pair")
ax.set_title("Ranking of overlapping bin pairs")

ax.invert_yaxis()

plt.tight_layout()