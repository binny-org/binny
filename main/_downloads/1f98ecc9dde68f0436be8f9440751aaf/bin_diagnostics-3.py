import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

def nested_dict_to_matrix(nested_dict):
    keys = sorted(nested_dict.keys())
    matrix = np.array(
        [[nested_dict[row_key][col_key] for col_key in keys] for row_key in keys],
        dtype=float,
    )
    return keys, matrix

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

overlap_min = tomo.cross_bin_stats(
    overlap={"method": "min", "unit": "percent", "normalize": True, "decimal_places": 3},
)["overlap"]

overlap_cosine = tomo.cross_bin_stats(
    overlap={"method": "cosine", "unit": "percent", "normalize": False, "decimal_places": 3},
)["overlap"]

overlap_js = tomo.cross_bin_stats(
    overlap={"method": "js", "unit": "fraction", "normalize": True, "decimal_places": 3},
)["overlap"]

overlap_hellinger = tomo.cross_bin_stats(
    overlap={"method": "hellinger", "unit": "fraction", "normalize": True, "decimal_places": 3},
)["overlap"]

overlap_tv = tomo.cross_bin_stats(
    overlap={"method": "tv", "unit": "fraction", "normalize": True, "decimal_places": 3},
)["overlap"]

metric_specs = [
    ("Min overlap [%]", overlap_min, "{:.1f}"),
    ("Cosine similarity [%]", overlap_cosine, "{:.1f}"),
    ("JS distance", overlap_js, "{:.2f}"),
    ("Hellinger distance", overlap_hellinger, "{:.2f}"),
    ("TV distance", overlap_tv, "{:.2f}"),
]

fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
axes = axes.ravel()

for ax, (title, metric_dict, fmt) in zip(axes[:-1], metric_specs, strict=True):
    keys, matrix = nested_dict_to_matrix(metric_dict)
    n_rows, n_cols = matrix.shape

    ax.imshow(
        matrix,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        alpha=0.6,
        interpolation="none",
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([f"{key + 1}" for key in keys])
    ax.set_yticklabels([f"{key + 1}" for key in keys])
    ax.set_xlabel("Tomographic bin")
    ax.set_ylabel("Tomographic bin")

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
                fontsize=13,
                color="k",
            )

axes[-1].axis("off")

plt.tight_layout()