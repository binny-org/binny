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

luminosity_functions = {
    r"Fiducial": LuminosityFunction(
        model="schechter",
        parameters={
            "phi_star": 3.0e-3,
            "m_star": -21.0,
            "alpha": -1.25,
        },
    ),
    r"Brighter $M_\star$": LuminosityFunction(
        model="schechter",
        parameters={
            "phi_star": 3.0e-3,
            "m_star": -21.6,
            "alpha": -1.25,
        },
    ),
    r"Steeper $\alpha$": LuminosityFunction(
        model="schechter",
        parameters={
            "phi_star": 3.0e-3,
            "m_star": -21.0,
            "alpha": -1.55,
        },
    ),
}

colors = cmr.take_cmap_colors(
    "viridis",
    len(luminosity_functions),
    cmap_range=(0.15, 0.85),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(7.0, 5.0))

for (label, lf), color in zip(luminosity_functions.items(), colors, strict=True):
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

    ax.plot(
        z,
        nz,
        color=color,
        linewidth=2.8,
        label=label,
    )

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Changing LF parameters at fixed cosmology")
ax.legend(frameon=False, loc="best")

plt.tight_layout()