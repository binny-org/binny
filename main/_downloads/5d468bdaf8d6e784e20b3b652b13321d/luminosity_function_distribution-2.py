import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from lfkit import LuminosityFunction
from binny.cosmology.ccl_wrappers import (
    comoving_volume_weight,
    luminosity_distance_mpc,
)
from binny import NZTomography

z = np.linspace(0.0, 3.5, 600)

cosmo = ccl.Cosmology(
    Omega_c=0.2607,
    Omega_b=0.049,
    h=0.6766,
    sigma8=0.8102,
    n_s=0.9665,
    transfer_function="bbks",
    matter_power_spectrum="linear",
)

lf = LuminosityFunction(
    model="schechter",
    parameters={
        "phi_star": 3.0e-3,
        "m_star": -21.0,
        "alpha": -1.25,
    },
)

lf_integral = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=lf,
    cosmo=cosmo,
    m_lim=25.3,
    m_bright=-26.0,
    n_m=512,
    volume_weight_fn=lambda z_eval: np.ones_like(z_eval),
    normalize=True,
)

volume = comoving_volume_weight(cosmo, z)
volume_scaled = volume / np.trapezoid(volume, z)

nz = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=lf,
    cosmo=cosmo,
    m_lim=25.3,
    m_bright=-26.0,
    n_m=512,
    normalize=True,
)

colors = cmr.take_cmap_colors(
    "viridis",
    5,
    cmap_range=(0.1, 0.9),
    return_fmt="hex",
)

color_lf = colors[0]
color_volume = colors[2]
color_nz = colors[4]

fig, ax1 = plt.subplots(figsize=(8.0, 5.2))
fig.patch.set_facecolor("white")

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 65))

line1, = ax1.plot(
    z,
    lf_integral,
    color=color_lf,
    linewidth=3.0,
    label="LF integral",
)
line2, = ax2.plot(
    z,
    volume_scaled,
    color=color_volume,
    linewidth=3.0,
    label="Volume weight",
)
line3, = ax3.plot(
    z,
    nz,
    color=color_nz,
    linewidth=3.0,
    label=r"$n(z)$",
)

ax1.set_xlabel("Redshift $z$")
ax1.set_ylabel("Normalized LF integral", color=color_lf)
ax2.set_ylabel("Normalized volume weight", color=color_volume)
ax3.set_ylabel(r"Normalized $n(z)$", color=color_nz)

ax1.tick_params(axis="y", colors=color_lf)
ax2.tick_params(axis="y", colors=color_volume)
ax3.tick_params(axis="y", colors=color_nz)

for ax in [ax1, ax2, ax3]:
    ax.tick_params(direction="in", axis="both", which="both")

lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]

legend = fig.legend(
    lines,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=3,
)

for line, text in zip(legend.get_lines(), legend.get_texts(), strict=True):
    text.set_color(line.get_color())

ax1.set_xlim(z.min(), z.max())
ax1.set_title("LF integral, volume weight, and final redshift distribution")