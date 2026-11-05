import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

nz_tophat = NZTomography.nz_model(
    "tophat",
    z,
    zmin=0.6,
    zmax=1.2,
    normalize=True,
)

color_tophat = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)[1]

fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.fill_between(
    z,
    0.0,
    nz_tophat,
    color=color_tophat,
    alpha=0.6,
    linewidth=0.0,
    zorder=10,
)
ax.plot(z, nz_tophat, color="k", linewidth=2.5, zorder=20)
ax.plot(z, np.zeros_like(z), color="k", linewidth=2.5, zorder=100)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Top-hat parent redshift distribution")

plt.tight_layout()