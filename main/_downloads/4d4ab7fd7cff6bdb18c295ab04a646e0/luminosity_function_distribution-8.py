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

print("z grid:")
print(z)
print()
print("LF-weighted n(z):")
print(nz)
print()
print("Shape:", nz.shape)
print("All non-negative:", bool(np.all(nz >= 0.0)))
print("Integral:", float(np.trapezoid(nz, z)))