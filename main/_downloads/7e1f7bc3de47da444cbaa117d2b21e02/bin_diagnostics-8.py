import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from binny import NZTomography

def nested_rect_dict_to_matrix(nested_dict):
    row_keys = sorted(nested_dict.keys())
    col_keys = sorted(nested_dict[row_keys[0]].keys())
    matrix = np.array(
        [[nested_dict[row_key][col_key] for col_key in col_keys] for row_key in row_keys],
        dtype=float,
    )
    return row_keys, col_keys, matrix

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

stats = lens.between_sample_stats(
    source,
    overlap={"method": "min", "unit": "percent", "normalize": True, "decimal_places": 3},
    interval_mass={"target_edges": target_edges, "unit": "percent", "decimal_places": 3},
    pearson={"normalize": True, "decimal_places": 3},
)

overlap_rows, overlap_cols, overlap_matrix = nested_rect_dict_to_matrix(stats["overlap"])
interval_rows, interval_cols, interval_matrix = nested_rect_dict_to_matrix(stats["interval_mass"])
pearson_rows, pearson_cols, pearson_matrix = nested_rect_dict_to_matrix(stats["pearson"])

base = plt.get_cmap("viridis")
colors = base(np.linspace(0.05, 0.95, 256))
colors[:, -1] = 0.6
cmap_transparent = ListedColormap(colors)

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

matrices = [
    (
        overlap_rows,
        overlap_cols,
        overlap_matrix,
        "Between-sample overlap",
        "Source bin",
        "Lens bin",
        "{:.1f}",
    ),
    (
        interval_rows,
        interval_cols,
        interval_matrix,
        "Source mass in lens intervals",
        "Lens nominal interval",
        "Source bin",
        "{:.1f}",
    ),
    (
        pearson_rows,
        pearson_cols,
        pearson_matrix,
        "Between-sample Pearson",
        "Source bin",
        "Lens bin",
        "{:.2f}",
    ),
]

for ax, (row_keys, col_keys, matrix, title, xlabel, ylabel, fmt) in zip(
    axes,
    matrices,
    strict=True,
):
    n_rows, n_cols = matrix.shape

    ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap_transparent,
        interpolation="none",
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([f"{key + 1}" for key in col_keys])
    ax.set_yticklabels([f"{key + 1}" for key in row_keys])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="k",
            )

plt.tight_layout()