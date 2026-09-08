import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

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

maglim = 25.0
r_psf_values = np.array([0.35, 0.50, 0.65, 0.80])
z_edges = np.linspace(0.0, 3.0, 41)

result = NZTomography.calibrate_psf_depth_from_mock(
    z_true=z_true,
    mag=mag,
    r_gal=r_gal,
    maglims=np.array([maglim]),
    r_psf_values=r_psf_values,
    area_deg2=100.0,
    z_edges=z_edges,
    selection_kind="sigmoid",
    normalize_nz=True,
)

colors = cmr.take_cmap_colors(
    "viridis",
    len(r_psf_values),
    cmap_range=(0.1, 0.9),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(8.0, 5.2))

for color, row in zip(colors, result["results"], strict=True):
    if np.isclose(row["maglim"], maglim):
        ax.stairs(
            row["nz"],
            z_edges,
            color=color,
            linewidth=3.0,
            label=rf"$R_{{\rm PSF}}={row['r_psf']:.2f}$",
        )

ax.set_xlabel(r"Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title(r"PSF-dependent source redshift distributions")
ax.legend(frameon=False)

plt.tight_layout()