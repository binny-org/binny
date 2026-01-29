import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# === Smail parameters (realistic shape) ===
z0 = 0.22
alpha = 1.5
beta = 2.0

# Limit z-range to remove long tail
z = np.linspace(0.01, 1.0, 500)


def smail_z(z, z0, alpha, beta):
    return (z / z0) ** beta * np.exp(-((z / z0) ** alpha))


# Compute and normalize Smail
N_z = smail_z(z, z0, alpha, beta)
N_z /= np.trapezoid(N_z, z)

# === Define Gaussians that fit Smail both in width and height ===
n_bins = 5
z_start, z_stop = 0.3, 0.7
bin_centers = np.linspace(z_start, z_stop, n_bins)
bin_width = (z_stop - z_start) / n_bins
gauss_sigmas = [bin_width / 2.5] * n_bins
fraction_of_smail = 0.95

# --- Binny-style colors (same as your tomography plots) ---
colors = cmr.take_cmap_colors("cmr.neon", n_bins, cmap_range=(0.25, 1.0), return_fmt="hex")

gauss_curves = []
for mu, sig in zip(bin_centers, gauss_sigmas, strict=False):
    G = norm.pdf(z, loc=mu, scale=sig)
    G /= G.max()
    smail_at_mu = np.interp(mu, z, N_z)
    G_scaled = fraction_of_smail * smail_at_mu * G
    gauss_curves.append(G_scaled)

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 5))

# Smail curve (black)
ax.plot(z, N_z, color="black", linewidth=2)

# Colored Gaussians (bins)
for g, color in zip(gauss_curves, colors, strict=True):
    ax.plot(z, g, color=color, linewidth=2)

# "binny" label
ax.text(
    0.85,
    -0.02,
    "binny",
    fontsize=32,
    fontweight="bold",
    ha="center",
    va="top",
    family="DejaVu Sans",
)

# Clean plot
ax.set_xlim(z.min(), z.max())
ax.set_ylim(-0.05, N_z.max() * 1.1)
ax.axis("off")

plt.tight_layout()
plt.show()
