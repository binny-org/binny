import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

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
r_psf_values = np.linspace(0.3, 1.0, 12)
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

psf = []
n_selected = []

for row in result["results"]:
    if np.isclose(row["maglim"], maglim):
        psf.append(row["r_psf"])
        n_selected.append(row["n_selected_arcmin2"])

color = cmr.take_cmap_colors(
    "viridis",
    1,
    cmap_range=(0.65, 0.65),
    return_fmt="hex",
)[0]

fig, ax = plt.subplots(figsize=(7.5, 5.0))

ax.plot(
    psf,
    n_selected,
    color="k",
    lw=2.0,
    zorder=1,
)
ax.scatter(
    psf,
    n_selected,
    s=70,
    color=to_rgba(color, 0.65),
    edgecolor="k",
    linewidth=2.0,
    zorder=2,
)

ax.set_xlabel(r"$R_{\rm PSF}$")
ax.set_ylabel(r"$n_{\rm selected}$ [arcmin$^{-2}$]")
ax.set_title(r"Selected source density versus PSF size")

plt.tight_layout()