import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from lfkit import LuminosityFunction
from binny import NZTomography

def gama_lfkit_evolving_schechter(
    absolute_mag,
    z,
    *,
    phi_star,
    m_star,
    alpha,
    q,
    p,
    z0=0.1,
):
    absolute_mag = np.asarray(absolute_mag)
    z = np.asarray(z)

    if absolute_mag.ndim > z.ndim:
        z = z[..., None]

    lf = LuminosityFunction.schechter(
        phi_star=phi_star * 10.0 ** (0.4 * p * z),
        m_star=m_star - q * (z - z0),
        alpha=alpha,
    )

    return lf.phi(absolute_mag)

z = np.linspace(0.0, 0.8, 500)

cosmo = ccl.Cosmology(
    Omega_c=0.2607,
    Omega_b=0.049,
    h=0.6766,
    sigma8=0.8102,
    n_s=0.9665,
    transfer_function="bbks",
    matter_power_spectrum="linear",
)

nz_blue = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=gama_lfkit_evolving_schechter,
    cosmo=cosmo,
    m_lim=19.8,
    m_bright=-24.0,
    n_m=512,
    normalize=True,
    phi_star=0.0038,
    m_star=-20.45,
    alpha=-1.49,
    q=0.8,
    p=2.9,
)

nz_red = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=gama_lfkit_evolving_schechter,
    cosmo=cosmo,
    m_lim=19.8,
    m_bright=-24.0,
    n_m=512,
    normalize=True,
    phi_star=0.0111,
    m_star=-20.34,
    alpha=-0.57,
    q=1.8,
    p=-1.2,
)

nz_all = NZTomography.nz_model(
    "luminosity_function",
    z,
    lf=gama_lfkit_evolving_schechter,
    cosmo=cosmo,
    m_lim=19.8,
    m_bright=-24.0,
    n_m=512,
    normalize=True,
    phi_star=0.94,
    m_star=-20.7,
    alpha=-1.23,
    q=0.7,
    p=1.8,
)

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.2, 0.8),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(7.4, 5.0))

ax.plot(z, nz_blue, color=colors[0], linewidth=3.0, label="Blue galaxies")
ax.fill_between(z, 0.0, nz_blue, color=colors[0], alpha=0.18, linewidth=0.0)

ax.plot(z, nz_red, color=colors[2], linewidth=3.0, label="Red galaxies")
ax.fill_between(z, 0.0, nz_red, color=colors[2], alpha=0.18, linewidth=0.0)

ax.plot(z, nz_all, color=colors[1], linewidth=3.0, label="All galaxies")
ax.fill_between(z, 0.0, nz_all, color=colors[1], alpha=0.18, linewidth=0.0)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Blue and red split LF modelling of the parent redshift distribution")
ax.legend(frameon=False, loc="best")

plt.tight_layout()