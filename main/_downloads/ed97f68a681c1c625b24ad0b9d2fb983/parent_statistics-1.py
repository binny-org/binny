import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

z = np.linspace(0.0, 2.0, 1000)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

result = NZTomography().build_predefined_bins(
    z=z,
    nz=nz,
    bins={0: nz},
)

moments = result.parent_stats(decimal_places=None)["moments"]

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

colors = dict(zip(("mean", "median", "mode"), colors, strict=True))

fig, ax = plt.subplots(figsize=(8.0, 4.8))

ax.plot(
    z,
    nz,
    color="k",
    linewidth=2.5,
    label=r"Parent $n(z)$",
)

for key in ("mean", "median", "mode"):
    ax.axvline(
        moments[key],
        color=colors[key],
        linestyle="-",
        linewidth=2.0,
        label=f"{key.capitalize()} = {moments[key]:.3f}",
    )

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Representative parent redshifts")
ax.legend(frameon=False)

plt.tight_layout()