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

maglim = 24.5
sel = mag <= maglim

result = NZTomography.calibrate_smail_from_mock(
    z_true=z_true,
    mag=mag,
    maglims=np.array([22.5, 23.0, 23.5, 24.0, 24.5]),
    area_deg2=5.0,
    infer_alpha_beta_from="deep_cut",
    alpha_beta_maglim=24.5,
    z_max=3.0,
)

alpha = result["alpha_beta_fit"]["params"]["alpha"]
beta = result["alpha_beta_fit"]["params"]["beta"]

z0_fit = result["z0_of_maglim"]["fit"]
if z0_fit["law"] == "linear":
    z0 = z0_fit["a"] * maglim + z0_fit["b"]
elif z0_fit["law"] == "poly2":
    z0 = z0_fit["c2"] * maglim**2 + z0_fit["c1"] * maglim + z0_fit["c0"]
else:
    raise ValueError(f"Unknown z0 law: {z0_fit['law']}")

z = np.linspace(0.0, 3.0, 600)
nz_fit = NZTomography.nz_model(
    "smail",
    z,
    z0=z0,
    alpha=alpha,
    beta=beta,
    normalize=True,
)

colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0, 1),
    return_fmt="hex"
)
_, c_hist, _, c_fit = colors

hist_fill = to_rgba(c_hist, 0.6)  # alpha applied only to fill
fit_fill = to_rgba(c_fit, 0.6)  # alpha only on fill

plt.figure(figsize=(8.0, 5.2))
plt.hist(
    z_true[sel],
    bins=20,
    range=(0.0, 3.0),
    density=True,
    edgecolor="k",
    linewidth=3,
    color=hist_fill,
    label="Mock sample",
)

# filled analytic model
plt.fill_between(
    z,
    0.0,
    nz_fit,
    color=fit_fill,
    edgecolor="k",
    linewidth=3.0,
    zorder=20,
    label="Fitted Smail model",
)

plt.xlabel("Redshift $z$")
plt.ylabel(r"Normalized $n(z)$")
plt.title("Mock redshift sample and calibrated Smail fit")
plt.legend(frameon=False)
plt.tight_layout()