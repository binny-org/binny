import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from lfkit import LuminosityFunction
from binny import NZTomography

z = np.linspace(0.0, 3.0, 500)

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

color = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)[-1]

fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.fill_between(
    z,
    0.0,
    nz,
    color=color,
    alpha=0.6,
    linewidth=0.0,
    zorder=10,
)
ax.plot(z, nz, color="k", linewidth=2.5, zorder=20)
ax.plot(z, np.zeros_like(z), color="k", linewidth=2.5, zorder=100)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("LF-weighted parent redshift distribution")

plt.tight_layout()