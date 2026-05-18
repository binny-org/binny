import numpy as np

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "gaussian",
    z,
    mu=1.0,
    sigma=0.25,
    normalize=True,
)

print("z grid:")
print(z)
print()
print("n(z) values:")
print(nz)
print()
print("Shape:", nz.shape)
print("All non-negative:", bool(np.all(nz >= 0.0)))