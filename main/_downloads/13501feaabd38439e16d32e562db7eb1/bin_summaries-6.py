import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from binny import NZTomography

z = np.linspace(0.0, 2.0, 1500)

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
)
shape_equidistant = tomo_equidistant.shape_stats(
    center_method="median",
    decimal_places=4,
)

keys = sorted(shape_equipopulated["per_bin"].keys())
x = np.arange(len(keys))
width = 0.36

values_equipopulated = [
    shape_equipopulated["per_bin"][key]["peaks"]["second_peak_ratio"] or 0.0
    for key in keys
]
values_equidistant = [
    shape_equidistant["per_bin"][key]["peaks"]["second_peak_ratio"] or 0.0
    for key in keys
]

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
    values_equipopulated,
    width=width,
    color=fill_eqpop,
    edgecolor="k",
    linewidth=2.5,
    label="Equipopulated",
)
plt.bar(
    x + width / 2,
    values_equidistant,
    width=width,
    color=fill_eqdist,
    edgecolor="k",
    linewidth=2.5,
    label="Equidistant",
)

plt.xticks(x, [f"{key+1}" for key in keys])
plt.xlabel("Tomographic bin")
plt.ylabel("Second peak / first peak")
plt.title("Secondary peak strength")
plt.legend(frameon=False)
plt.tight_layout()