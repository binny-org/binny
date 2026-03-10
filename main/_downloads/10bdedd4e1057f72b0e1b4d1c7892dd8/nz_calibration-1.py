import numpy as np

from binny import NZTomography

rng = np.random.default_rng(42)
n_gal = 30000

z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

# Make magnitudes loosely correlated with redshift, with scatter
mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

# Define magnitude limits for calibration
maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

result = NZTomography.calibrate_smail_from_mock(
    z_true=z_true,
    mag=mag,
    maglims=maglims,
    area_deg2=100.0,
    infer_alpha_beta_from="deep_cut",
    alpha_beta_maglim=24.5,
    z_max=3.0,
)

print("Calibration succeeded:", result["ok"])
print()

print("Inferred Smail shape parameters:")
print(result["alpha_beta_fit"])
print()

print("Calibrated z0(maglim) relation:")
print(result["z0_of_maglim"])
print()

print("Calibrated ngal(maglim) relation:")
print(result["ngal_of_maglim"])