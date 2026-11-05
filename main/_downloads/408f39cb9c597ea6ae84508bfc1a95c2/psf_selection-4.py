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
r_psf = 0.65
z_edges = np.linspace(0.0, 3.0, 41)
z_fine = np.linspace(0.0, 3.0, 600)

colors = cmr.take_cmap_colors(
    "viridis",
    2,
    cmap_range=(0.2, 0.8),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(8.0, 5.2))

for color, selection_kind in zip(colors, ["hard", "sigmoid"], strict=True):
    result = NZTomography.calibrate_psf_depth_from_mock(
        z_true=z_true,
        mag=mag,
        r_gal=r_gal,
        maglims=np.array([maglim]),
        r_psf_values=np.array([r_psf]),
        area_deg2=100.0,
        z_edges=z_edges,
        selection_kind=selection_kind,
        normalize_nz=True,
    )

    row = result["results"][0]

    smail_result = NZTomography.fit_smail_from_mock(
        row["z"],
        weights=row["weights"],
        z_max=3.0,
    )

    if not smail_result["ok"]:
        continue

    params = smail_result["params"]

    nz_fit = NZTomography.nz_model(
        "smail",
        z_fine,
        z0=params["z0"],
        alpha=params["alpha"],
        beta=params["beta"],
        normalize=True,
    )

    ax.stairs(
        row["nz"],
        z_edges,
        color=color,
        linewidth=1.5,
        alpha=0.35,
    )

    ax.plot(
        z_fine,
        nz_fit,
        color=color,
        lw=3.0,
        label=selection_kind,
    )

ax.set_xlabel(r"Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title(r"Hard-cut and sigmoid PSF-selection Smail fits")
ax.legend(frameon=False)

plt.tight_layout()