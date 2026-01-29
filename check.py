import traceback
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from binny.api.nz_tomography import NZTomography


def plot_parent_and_bins(
    z,
    nz,
    bins,
    *,
    title="NZTomography",
    ax=None,
    color=None,
    label_prefix="",
    linestyles=("-", "--", ":"),
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created = True

    # ax.plot(z, nz, linewidth=2, color="k", label="parent n(z)")

    for j, k in enumerate(sorted(bins.keys())):
        ls = linestyles[j % len(linestyles)]
        ax.plot(
            z,
            np.asarray(bins[k], float),
            linewidth=2,
            color=color,  # <-- same color for ALL bins in this run
            linestyle=ls,  # <-- differ bins by linestyle
            label=f"{label_prefix}bin {k}",
        )

    ax.set_xlabel("z")
    ax.set_ylabel("n(z)")
    ax.set_title(title)
    ax.legend()

    if created:
        plt.show()


t = NZTomography()
z = np.linspace(0, 3, 1000)
nz = t.nz_model("smail", z, z0=0.26, alpha=2.0, beta=0.94, normalize=True)

base_spec = {
    "kind": "photoz",
    "nz": {"model": "arrays"},
    "bins": {"scheme": "equidistant", "n_bins": 3},
}

runs = [
    {
        "outlier_frac": [0.10, 0.10, 0.10],
        "outlier_scatter_scale": [0.30, 0.30, 0.30],
        "outlier_mean_scale": [1.0, 1.0, 1.0],
        "outlier_mean_offset": [0.0, 0.0, 0.0],
    },
    {
        "outlier_frac": [0.15, 0.3, 0.2],
        "outlier_scatter_scale": [0.30, 0.30, 0.30],
        "outlier_mean_scale": [1.05, 1.10, 1.15],
        "outlier_mean_offset": [0.0, 0.0, 0.0],
    },
    {
        "outlier_frac": [0.2, 0.15, 0.12],
        "outlier_scatter_scale": [0.30, 0.30, 0.30],
        "outlier_mean_scale": [0.90, 0.85, 0.80],
        "outlier_mean_offset": [0.0, 0.0, 0.0],
    },
]


colors = ["C0", "C1", "C2"]

fig, ax = plt.subplots(figsize=(8, 4))

for i, (u, c) in enumerate(zip(runs, colors, strict=False), start=1):
    spec = deepcopy(base_spec)
    spec["uncertainties"] = u
    try:
        payload = t.build_bins(z=z, nz=nz, tomo_spec=spec)
        print("payload keys:", payload.keys())
    except Exception as e:
        print("EXCEPTION:", repr(e))
        traceback.print_exc()
        continue

    plot_parent_and_bins(
        z,
        nz,
        payload["bins"],
        ax=ax,
        color=c,
        label_prefix=f"run {i}: ",
        title="Custom photoz bins (3 runs overlaid)",
    )

plt.tight_layout()
plt.show()
