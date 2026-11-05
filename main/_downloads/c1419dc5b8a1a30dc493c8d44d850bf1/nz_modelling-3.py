import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

nz_standard = NZTomography.nz_model(
    "smail",
    z,
    z0=0.28,
    alpha=2.0,
    beta=1.5,
    normalize=True,
)

nz_shifted = NZTomography.nz_model(
    "shifted_smail",
    z,
    z0=0.28,
    alpha=2.0,
    beta=1.5,
    z_shift=0.25,
    normalize=True,
)

colors = cmr.take_cmap_colors(
    "viridis_r",
    4,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)
color_standard, _, color_shifted, _ = colors

fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.fill_between(
    z,
    0.0,
    nz_standard,
    color=color_standard,
    alpha=0.6,
    linewidth=0.0,
    zorder=10,
    label="Smail",
)
ax.plot(z, nz_standard, color="k", linewidth=2.5, zorder=20)

ax.fill_between(
    z,
    0.0,
    nz_shifted,
    color=color_shifted,
    alpha=0.6,
    linewidth=0.0,
    zorder=11,
    label="Shifted Smail",
)
ax.plot(z, nz_shifted, color="k", linewidth=2.5, zorder=21)
ax.plot(z, np.zeros_like(z), color="k", linewidth=2.5, zorder=100)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Standard vs shifted Smail distribution")
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()