import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny.nz.psf_selection import shear_selection_weight

r_gal = np.linspace(0.05, 1.5, 500)
r_psf = 0.65

hard = shear_selection_weight(
    r_gal,
    r_psf=r_psf,
    r_min=0.3,
    kind="hard",
)

sigmoid = shear_selection_weight(
    r_gal,
    r_psf=r_psf,
    r_min=0.3,
    kind="sigmoid",
    width=0.05,
)

colors = cmr.take_cmap_colors(
    "viridis",
    2,
    cmap_range=(0.2, 0.8),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(7.5, 5.0))

ax.plot(
    r_gal,
    hard,
    color=colors[0],
    lw=3,
    ls="-",
    label="hard cut",
)

ax.plot(
    r_gal,
    sigmoid,
    color=colors[1],
    lw=3.0,
    label="sigmoid selection",
)

ax.set_xlabel(r"Galaxy size $R_{\rm gal}$")
ax.set_ylabel("Selection weight")
ax.set_title(r"PSF-dependent shear-selection weight")
ax.legend(frameon=False)

plt.tight_layout()