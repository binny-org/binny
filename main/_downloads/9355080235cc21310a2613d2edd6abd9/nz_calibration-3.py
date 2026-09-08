import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import cmasher as cmr

from binny import NZTomography

rng = np.random.default_rng(42)

n_gal = 30000

z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

result = NZTomography.calibrate_smail_from_mock(
    z_true=z_true,
    mag=mag,
    maglims=maglims,
    area_deg2=5.0,
    infer_alpha_beta_from="deep_cut",
    alpha_beta_maglim=24.5,
    z_max=3.0,
)

z0_points = result["z0_of_maglim"]["points"]
z0_fit = result["z0_of_maglim"]["fit"]

ngal_points = result["ngal_of_maglim"]["points"]
ngal_fit = result["ngal_of_maglim"]["fit"]

mfit = np.linspace(maglims.min(), maglims.max(), 200)

if z0_fit["law"] == "linear":
    z0_curve = z0_fit["a"] * mfit + z0_fit["b"]
elif z0_fit["law"] == "poly2":
    z0_curve = z0_fit["c2"] * mfit**2 + z0_fit["c1"] * mfit + z0_fit["c0"]
else:
    raise ValueError(f"Unknown z0 law: {z0_fit['law']}")

if ngal_fit["law"] == "linear":
    ngal_curve = ngal_fit["p"] * mfit + ngal_fit["q"]
elif ngal_fit["law"] == "loglinear":
    ngal_curve = 10.0 ** (ngal_fit["s"] * mfit + ngal_fit["t"])
else:
    raise ValueError(f"Unknown ngal law: {ngal_fit['law']}")

cmap = "viridis"
c1 = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.15, 0.35))[1]
c2 = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.65, 0.85))[1]
fill1 = to_rgba(c1, 0.6)
fill2 = to_rgba(c2, 0.6)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

axes[0].plot(mfit, z0_curve, lw=3, color=c1, alpha=0.6)
axes[0].scatter(
    z0_points["maglim"],
    z0_points["z0"],
    s=150,
    facecolor=fill1,
    edgecolors="k",
    linewidth=2.0,
    zorder=20,
)
axes[0].set_xlabel(r"Limiting magnitude $m_{\rm lim}$")
axes[0].set_ylabel(r"Fitted $z_0$")
axes[0].set_title(r"Calibrated $z_0(m_{\rm lim})$")

axes[1].plot(mfit, ngal_curve, lw=3, color=c2, alpha=0.6)
axes[1].scatter(
    ngal_points["maglim"],
    ngal_points["ngal_arcmin2"],
    s=150,
    facecolor=fill2,
    edgecolors="k",
    linewidth=2.0,
    zorder=20,
)
axes[1].set_xlabel(r"Limiting magnitude $m_{\rm lim}$")
axes[1].set_ylabel(r"$n_{\rm gal}$ [arcmin$^{-2}$]")
axes[1].set_title(r"Calibrated $n_{\rm gal}(m_{\rm lim})$")

plt.tight_layout()