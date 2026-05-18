import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from binny import NZTomography

z = np.linspace(0.0, 2.5, 600)

lens_nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.18,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

source_nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.32,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

lens_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
    "uncertainties": {
        "scatter_scale": 0.03,
        "mean_offset": 0.00,
        "outlier_frac": 0.01,
        "outlier_scatter_scale": 0.10,
        "outlier_mean_offset": 0.03,
    },
    "normalize_bins": True,
}

source_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "scatter_scale": 0.06,
        "mean_offset": 0.01,
        "outlier_frac": 0.04,
        "outlier_scatter_scale": 0.25,
        "outlier_mean_offset": 0.06,
    },
    "normalize_bins": True,
}

lens = NZTomography()
lens_result = lens.build_bins(
    z=z,
    nz=lens_nz,
    tomo_spec=lens_spec,
    include_tomo_metadata=True,
)

source = NZTomography()
source.build_bins(
    z=z,
    nz=source_nz,
    tomo_spec=source_spec,
    include_tomo_metadata=True,
)

target_edges = lens_result.tomo_meta["bins"]["bin_edges"]

interval_mass = lens.between_sample_stats(
    source,
    interval_mass={
        "target_edges": target_edges,
        "unit": "percent",
        "decimal_places": 3,
    },
)["interval_mass"]

source_keys = sorted(interval_mass.keys())
lens_keys = sorted(interval_mass[source_keys[0]].keys())

x = np.arange(len(source_keys))

colors = cmr.take_cmap_colors(
    "viridis",
    len(lens_keys),
    cmap_range=(0.1, 0.9),
    return_fmt="hex",
)
fill_colors = [to_rgba(color, 0.6) for color in colors]

bottoms = np.zeros(len(source_keys))

fig, ax = plt.subplots(figsize=(8.2, 5.0))

for fill_color, lens_key in zip(fill_colors, lens_keys, strict=True):
    values = np.array(
        [interval_mass[source_key][lens_key] for source_key in source_keys],
        dtype=float,
    )
    ax.bar(
        x,
        values,
        bottom=bottoms,
        color=fill_color,
        edgecolor="k",
        linewidth=2.0,
        label=f"Lens interval {lens_key + 1}",
    )
    bottoms += values

ax.set_xticks(x)
ax.set_xticklabels([f"Source bin {key + 1}" for key in source_keys])
ax.set_xlabel("Source bin")
ax.set_ylabel("Percent of source-bin mass")
ax.set_title("Source-bin mass across lens intervals")
ax.legend(frameon=True, loc= "center left")

plt.tight_layout()