import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from lfkit import LuminosityFunction
from binny import NZTomography

z = np.linspace(0.0, 3.0, 500)

cosmologies = {
    r"$\Omega_m = 0.25$": ccl.Cosmology(
        Omega_c=0.201,
        Omega_b=0.049,
        h=0.6766,
        sigma8=0.8102,
        n_s=0.9665,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    ),
    r"$\Omega_m = 0.31$": ccl.Cosmology(
        Omega_c=0.2607,
        Omega_b=0.049,
        h=0.6766,
        sigma8=0.8102,
        n_s=0.9665,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    ),
    r"$\Omega_m = 0.37$": ccl.Cosmology(
        Omega_c=0.321,
        Omega_b=0.049,
        h=0.6766,
        sigma8=0.8102,
        n_s=0.9665,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    ),
}

lf = LuminosityFunction(
    model="schechter",
    parameters={
        "phi_star": 3.0e-3,
        "m_star": -21.0,
        "alpha": -1.25,
    },
)

colors = cmr.take_cmap_colors(
    "viridis",
    len(cosmologies),
    cmap_range=(0.1, 0.9),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(7.0, 5.0))

for (label, cosmo), color in zip(cosmologies.items(), colors, strict=True):
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
ax.set_title("Changing cosmology at fixed LF")
ax.legend(frameon=False, loc="best")

plt.tight_layout()