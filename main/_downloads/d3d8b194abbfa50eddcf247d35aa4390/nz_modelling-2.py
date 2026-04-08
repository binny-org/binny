import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

z = np.linspace(0.0, 2.0, 500)

z0 = 0.28
alpha = 2.0
beta = 1.5

nz_smail = NZTomography.nz_model(
    "smail",
    z,
    z0=z0,
    alpha=alpha,
    beta=beta,
    normalize=True,
)

color_smail = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)[-1]

fig, ax = plt.subplots(figsize=(7, 5))

ax.fill_between(
    z,
    0.0,
    nz_smail,
    color=color_smail,
    alpha=0.6,
    linewidth=0.0,
    zorder=10,
)
ax.plot(z, nz_smail, color="k", linewidth=2.5, zorder=20)
ax.plot(z, np.zeros_like(z), color="k", linewidth=2.5, zorder=100)

formula_text = (
    r"$n(z)\propto\left(\frac{z}{z_0}\right)^{\alpha}"
    r"\exp\!\left[-\left(\frac{z}{z_0}\right)^{\beta}\right]$"
)

param_text = (
    r"$z_0 = {:.2f}$" "\n"
    r"$\alpha = {:.1f}$" "\n"
    r"$\beta = {:.1f}$"
).format(z0, alpha, beta)

ax.text(
    0.55,
    0.95,
    formula_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
)

ax.text(
    0.95,
    0.84,
    param_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Smail parent redshift distribution")

plt.tight_layout()