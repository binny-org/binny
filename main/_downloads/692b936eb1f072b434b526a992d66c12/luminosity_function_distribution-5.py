import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from lfkit import LuminosityFunction
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

magnitude_limits = [24.5, 25.3, 26.0]

colors = cmr.take_cmap_colors(
    "viridis",
    len(magnitude_limits),
    cmap_range=(0.15, 0.85),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(7.0, 5.0))

for m_lim, color in zip(magnitude_limits, colors, strict=True):
    nz = NZTomography.nz_model(
        "luminosity_function",
        z,
        lf=lf,
        cosmo=cosmo,
        m_lim=m_lim,
        m_bright=-26.0,
        n_m=512,
        normalize=True,
    )

    ax.plot(
        z,
        nz,
        color=color,
        linewidth=2.8,
        label=rf"$m_{{\rm lim}} = {m_lim:.1f}$",
    )

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Magnitude-limit dependence")
ax.legend(frameon=False, loc="best")

plt.tight_layout()