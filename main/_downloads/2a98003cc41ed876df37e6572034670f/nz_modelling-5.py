import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography

def plot_parent_models(ax, z, model_curves, title):
    colors = cmr.take_cmap_colors(
        "viridis",
        3,
        cmap_range=(0.0, 1.0),
        return_fmt="hex",
    )

    for i, ((label, nz_values), color) in enumerate(
        zip(model_curves, colors, strict=True)
    ):
        ax.fill_between(
            z,
            0.0,
            nz_values,
            color=color,
            alpha=0.6,
            linewidth=0.0,
            zorder=10 + i,
            label=label,
        )
        ax.plot(
            z,
            nz_values,
            color="k",
            linewidth=1.8,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.5, zorder=1000)

    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n(z)$")
    ax.legend(frameon=False, loc="best")

z = np.linspace(0.0, 1.5, 500)

panel1_models = [
    (
        "Smail",
        NZTomography.nz_model(
            "smail",
            z,
            z0=0.28,
            alpha=2.0,
            beta=1.5,
            normalize=True,
        ),
    ),
    (
        "Gamma",
        NZTomography.nz_model(
            "gamma",
            z,
            k=1.5,
            theta=0.28,
            normalize=True,
        ),
    ),
    (
        "Schechter",
        NZTomography.nz_model(
            "schechter",
            z,
            z0=0.2,
            alpha=2.0,
            normalize=True,
        ),
    ),
]


panel2_models = [
    (
        "Gaussian",
        NZTomography.nz_model(
            "gaussian",
            z,
            mu=0.9,
            sigma=0.22,
            normalize=True,
        ),
    ),
    (
        "Gaussian mixture",
        NZTomography.nz_model(
            "gaussian_mixture",
            z,
            mus=np.array([0.55, 1.25]),
            sigmas=np.array([0.12, 0.20]),
            weights=np.array([0.45, 0.55]),
            normalize=True,
        ),
    ),
    (
        "Top-hat",
        NZTomography.nz_model(
            "tophat",
            z,
            zmin=0.6,
            zmax=1.2,
            normalize=True,
        ),
    ),
]


fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)

plot_parent_models(axes[0], z, panel1_models, "Survey-like parent $n(z)$ models")
plot_parent_models(axes[1], z, panel2_models, "Toy and mixture parent $n(z)$ models")

plt.tight_layout()