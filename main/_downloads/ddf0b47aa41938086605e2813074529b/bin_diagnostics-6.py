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
x = np.arange(len(keys))

colors = cmr.take_cmap_colors(
    "viridis",
    len(keys),
    cmap_range=(0.1, 0.9),
    return_fmt="hex",
)
fill_colors = [to_rgba(color, 0.6) for color in colors]

bottoms = np.zeros(len(keys))

fig, ax = plt.subplots(figsize=(8.2, 5.0))

for fill_color, target_key in zip(fill_colors, keys, strict=True):
    values = [leakage[source_key][target_key] for source_key in keys]
    ax.bar(
        x,
        values,
        bottom=bottoms,
        color=fill_color,
        edgecolor="k",
        linewidth=2.0,
        label=f"Nominal bin {target_key + 1}",
    )
    bottoms += np.array(values)

ax.set_xticks(x)
ax.set_xticklabels([f"Input bin {key + 1}" for key in keys])
ax.set_xlabel("Input bin")
ax.set_ylabel("Input-bin mass [%]")
ax.set_title("Leakage composition")
ax.legend(frameon=True, loc="center left")

plt.tight_layout()