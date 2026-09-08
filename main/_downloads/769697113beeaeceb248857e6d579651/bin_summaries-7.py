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

common_uncertainties = {
    "scatter_scale": [0.010, 0.012, 0.015, 0.018],
    "mean_offset": 0.0,
    "outlier_frac": [0.02, 0.05, 0.15, 0.26],
    "outlier_scatter_scale": [0.008, 0.010, 0.012, 0.015],
    "outlier_mean_offset": [0.35, 0.40, 0.45, 0.50],
}

equipopulated_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equidistant", "n_bins": 4, "range": (0.2, 1.2)},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equipopulated_edges = [0.20, 0.33, 0.49, 0.72, 1.20]
equidistant_edges = [0.20, 0.45, 0.70, 0.95, 1.20]

tomo_equipopulated = NZTomography()
tomo_equipopulated.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equipopulated_spec,
    include_tomo_metadata=True,
)

tomo_equidistant = NZTomography()
tomo_equidistant.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equidistant_spec,
    include_tomo_metadata=True,
)

shape_equipopulated = tomo_equipopulated.shape_stats(
    center_method="median",
    decimal_places=4,
    bin_edges=equipopulated_edges,
)

shape_equidistant = tomo_equidistant.shape_stats(
    center_method="median",
    decimal_places=4,
    bin_edges=equidistant_edges,
)

fractions_equipopulated = shape_equipopulated["in_range_fraction"]
fractions_equidistant = shape_equidistant["in_range_fraction"]

keys = sorted(fractions_equipopulated.keys())
x = np.arange(len(keys))
width = 0.36

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

_, c_eqpop, c_eqdist = colors
fill_eqpop = to_rgba(c_eqpop, 0.6)
fill_eqdist = to_rgba(c_eqdist, 0.6)

plt.figure(figsize=(7.8, 4.8))
plt.bar(
    x - width / 2,
    [100.0 * fractions_equipopulated[key] for key in keys],
    width=width,
    color=fill_eqpop,
    edgecolor="k",
    linewidth=2.5,
    label="Equipopulated",
)
plt.bar(
    x + width / 2,
    [100.0 * fractions_equidistant[key] for key in keys],
    width=width,
    color=fill_eqdist,
    edgecolor="k",
    linewidth=2.5,
    label="Equidistant",
)

plt.xticks(x, [f"{key+1}" for key in keys])
plt.xlabel("Tomographic bin")
plt.ylabel("In-range fraction [%]")
plt.title("Fraction of each bin inside its nominal range")
plt.legend(frameon=True)
plt.tight_layout()