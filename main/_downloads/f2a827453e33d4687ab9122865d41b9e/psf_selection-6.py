import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

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

maglims = np.array([23.0, 23.5, 24.0, 24.5, 25.0])
r_psf_values = np.array([0.35, 0.50, 0.65, 0.80, 0.95])
z_edges = np.linspace(0.0, 3.0, 41)

result = NZTomography.calibrate_psf_depth_from_mock(
    z_true=z_true,
    mag=mag,
    r_gal=r_gal,
    maglims=maglims,
    r_psf_values=r_psf_values,
    area_deg2=100.0,
    z_edges=z_edges,
    selection_kind="sigmoid",
    normalize_nz=True,
)

grid = np.full(
    (len(r_psf_values), len(maglims)),
    np.nan,
)

for row in result["results"]:
    i = np.where(np.isclose(r_psf_values, row["r_psf"]))[0][0]
    j = np.where(np.isclose(maglims, row["maglim"]))[0][0]
    grid[i, j] = row["n_selected_arcmin2"]

base = plt.get_cmap("viridis")
colors = base(np.linspace(0.05, 0.95, 256))
colors[:, -1] = 0.6
cmap_transparent = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(7.8, 5.2))

im = ax.imshow(
    grid,
    origin="lower",
    aspect="auto",
    cmap=cmap_transparent,
    interpolation="none",
)

n_rows, n_cols = grid.shape

ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))
ax.set_xticklabels([f"{m:.1f}" for m in maglims])
ax.set_yticklabels([f"{r:.2f}" for r in r_psf_values])

ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(n_rows):
    for j in range(n_cols):
        ax.text(
            j,
            i,
            f"{grid[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            color="k",
        )

ax.set_xlabel(r"Limiting magnitude $m_{\rm lim}$")
ax.set_ylabel(r"$R_{\rm PSF}$")
ax.set_title(r"$n_{\rm selected}$ across depth and PSF size")

plt.colorbar(
    im,
    ax=ax,
    label=r"$n_{\rm selected}$ [arcmin$^{-2}$]",
)

plt.tight_layout()