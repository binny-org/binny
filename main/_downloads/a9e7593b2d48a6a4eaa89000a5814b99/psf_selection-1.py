import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

rng = np.random.default_rng(42)

n_gal = 50000

z_true = rng.gamma(shape=2.0, scale=0.45, size=n_gal)
z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

mag = 22.0 + 2.1 * z_true + rng.normal(
    0.0,
    0.4,
    size=z_true.size,
)

r_gal = (
    0.7 / (1.0 + z_true)
    + rng.normal(0.0, 0.08, size=z_true.size)
)
r_gal = np.clip(r_gal, 0.05, None)

color = cmr.take_cmap_colors(
    "viridis",
    1,
    cmap_range=(0.45, 0.45),
    return_fmt="hex",
)[0]

fig, ax = plt.subplots(figsize=(7.5, 5.0))

ax.scatter(
    z_true,
    r_gal,
    s=6,
    color=to_rgba(color, 0.25),
    edgecolors="none",
)

ax.set_xlabel(r"True redshift $z$")
ax.set_ylabel(r"Galaxy size $R_{\rm gal}$")
ax.set_title("Synthetic mock galaxy sizes")

plt.tight_layout()