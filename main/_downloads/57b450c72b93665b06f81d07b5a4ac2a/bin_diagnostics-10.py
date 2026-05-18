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
lens.build_bins(
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

pair_list = lens.between_sample_stats(
    source,
    pairs={
        "method": "min",
        "unit": "percent",
        "threshold": 0.0,
        "direction": "high",
        "normalize": True,
        "decimal_places": 3,
    },
)["correlations"]

labels = [f"L{i+1} - S{j+1}" for i, j, _ in pair_list]
values = np.array([value for _, _, value in pair_list])
y = np.arange(len(labels))

colors = cmr.take_cmap_colors(
    "viridis",
    len(labels),
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)
fill_colors = [to_rgba(color, 0.6) for color in colors]

fig, ax = plt.subplots(figsize=(8.2, 7))

ax.barh(
    y,
    values,
    color=fill_colors,
    edgecolor="k",
    linewidth=2.5,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Between-sample min overlap [%]")
ax.set_ylabel("Lens–source pair")
ax.set_title("Ranking of cross-sample overlapping pairs")

ax.invert_yaxis()

plt.tight_layout()