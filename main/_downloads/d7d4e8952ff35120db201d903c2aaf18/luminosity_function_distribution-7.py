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

nz_unnormalized = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=lf,
    cosmo=cosmo,
    m_lim=25.3,
    m_bright=-26.0,
    n_m=512,
    normalize=False,
)

nz_normalized = NZTomography.nz_model(
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
    2,
    cmap_range=(0.15, 0.85),
    return_fmt="hex",
)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

axes[0].plot(
    z,
    nz_unnormalized,
    color=colors[0],
    linewidth=2.8,
)
axes[0].set_xlabel("Redshift $z$")
axes[0].set_ylabel(r"Unnormalized $dN/dz$")
axes[0].set_title("Unnormalized output")
axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

axes[1].plot(
    z,
    nz_normalized,
    color=colors[1],
    linewidth=2.8,
)
axes[1].set_xlabel("Redshift $z$")
axes[1].set_ylabel(r"Normalized $n(z)$")
axes[1].set_title("Normalized output")

plt.tight_layout()