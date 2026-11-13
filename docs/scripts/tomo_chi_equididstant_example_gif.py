"""Animate equal-chi tomography against conventional equal-redshift binning."""

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from binny.nz import nz_model
from binny.tomography import build_tomography_from_spec


def replace_fill(ax, old_fill, x, y, color, alpha=0.6):
    """Replace an existing filled curve."""
    old_fill.remove()
    return ax.fill_between(
        x,
        0.0,
        y,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=10,
    )


def ordered_curves(bin_dict):
    """Return tomographic curves ordered by bin index."""
    keys = sorted(bin_dict)
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def top_hat_window(z, zmin, zmax, include_right=False):
    """Construct a top-hat selection window."""
    if include_right:
        return ((z >= zmin) & (z <= zmax)).astype(float)
    return ((z >= zmin) & (z < zmax)).astype(float)


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_chi_equidistant_binning.gif"

z = np.linspace(0.0, 2.0, 500)

nz = nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

cosmo = ccl.CosmologyVanillaLCDM()

n_bins = 4
bin_range = (0.2, 1.2)

chi_min = ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + bin_range[0]))
chi_max = ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + bin_range[1]))

chi_edges = np.linspace(chi_min, chi_max, n_bins + 1)

z_fine = np.linspace(0.0, 2.0, 10000)
chi_fine = ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + z_fine))

z_edges = np.interp(chi_edges, chi_fine, z_fine)
equal_z_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

hard_curves = []

for i in range(n_bins):
    window = top_hat_window(
        z,
        z_edges[i],
        z_edges[i + 1],
        include_right=(i == n_bins - 1),
    )
    hard_curves.append(nz * window)

specz_spec = {
    "kind": "specz",
    "bins": {
        "edges": z_edges.tolist(),
    },
    "normalize_bins": True,
}

photoz_spec = {
    "kind": "photoz",
    "bins": {
        "edges": z_edges.tolist(),
    },
    "uncertainties": {
        "scatter_scale": 0.05,
    },
    "normalize_bins": True,
}

equal_z_specz_spec = {
    "kind": "specz",
    "bins": {
        "edges": equal_z_edges.tolist(),
    },
    "normalize_bins": True,
}

equal_z_photoz_spec = {
    "kind": "photoz",
    "bins": {
        "edges": equal_z_edges.tolist(),
    },
    "uncertainties": {
        "scatter_scale": 0.05,
    },
    "normalize_bins": True,
}

specz_result = build_tomography_from_spec(
    z=z,
    nz=nz,
    tomo_spec=specz_spec,
)

photoz_result = build_tomography_from_spec(
    z=z,
    nz=nz,
    tomo_spec=photoz_spec,
)

equal_z_specz_result = build_tomography_from_spec(
    z=z,
    nz=nz,
    tomo_spec=equal_z_specz_spec,
)

equal_z_photoz_result = build_tomography_from_spec(
    z=z,
    nz=nz,
    tomo_spec=equal_z_photoz_spec,
)

specz_curves = ordered_curves(specz_result.bins)
photoz_curves = ordered_curves(photoz_result.bins)

equal_z_specz_curves = ordered_curves(equal_z_specz_result.bins)
equal_z_photoz_curves = ordered_curves(equal_z_photoz_result.bins)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

reveal_vals = np.linspace(z.min(), z.max(), 50)

all_curves = [
    hard_curves,
    specz_curves,
    photoz_curves,
]

reference_curves = [
    None,
    equal_z_specz_curves,
    equal_z_photoz_curves,
]

ymax = [
    1.08 * np.max(nz),
    1.08 * max(np.max(curve) for curve in specz_curves + equal_z_specz_curves),
    1.08 * max(np.max(curve) for curve in photoz_curves + equal_z_photoz_curves),
]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(9.0, 10.0),
    sharex=True,
)

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 15
lw = 2.0

titles = [
    "Parent distribution with comoving-distance cuts",
    "Spectroscopic tomographic bins",
    "Photometric tomographic bins",
]

for ax, title, ylim in zip(axes, titles, ymax, strict=True):
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, ylim)

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=lw,
        length=6,
        labelsize=tick_fs,
    )

    ax.grid(False)
    ax.plot(
        z,
        np.zeros_like(z),
        color="k",
        linewidth=lw,
        zorder=100,
    )

axes[0].set_ylabel(r"$n(z)$", fontsize=label_fs)
axes[1].set_ylabel(r"$n_i(z)$", fontsize=label_fs)
axes[2].set_ylabel(r"$n_i(z)$", fontsize=label_fs)
axes[2].set_xlabel("Redshift $z$", fontsize=label_fs)

axes[0].plot(
    z,
    nz,
    color="k",
    linewidth=lw,
    zorder=20,
)

fills = []
lines = []
reference_lines = []

for panel, ax in enumerate(axes):
    panel_fills = []
    panel_lines = []
    panel_reference_lines = []

    for color in colors:
        fill = ax.fill_between(
            z,
            0.0,
            np.zeros_like(z),
            color=color,
            alpha=0.6,
            linewidth=0.0,
            zorder=10,
        )

        (line,) = ax.plot(
            [],
            [],
            color="k",
            linewidth=lw,
            zorder=25,
        )

        panel_fills.append(fill)
        panel_lines.append(line)

        if panel > 0:
            (reference_line,) = ax.plot(
                [],
                [],
                color="0.55",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                zorder=5,
            )
            panel_reference_lines.append(reference_line)

    fills.append(panel_fills)
    lines.append(panel_lines)
    reference_lines.append(panel_reference_lines)

annotations = []

annotation_positions = [
    (0.54, 0.88),
    (0.54, 0.93),
    (0.54, 0.93),
]

for ax, (xpos, ypos) in zip(axes, annotation_positions, strict=True):
    annotation = ax.text(
        xpos,
        ypos,
        "",
        transform=ax.transAxes,
        fontsize=annot_fs,
        ha="left",
        va="top",
    )
    annotations.append(annotation)

legend_handles = [
    Line2D(
        [0],
        [0],
        color="k",
        linewidth=2.0,
        label=r"Equal $\Delta\chi$",
    ),
    Line2D(
        [0],
        [0],
        color="0.55",
        linestyle="--",
        linewidth=2.0,
        label=r"Equal $\Delta z$",
    ),
]

for ax in axes[1:]:
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=12,
        frameon=False,
    )


def draw_frame(zcut):
    """Update the selected and reference tomographic curves."""
    for panel, curves in enumerate(all_curves):
        for i in range(n_bins):
            y = np.where(
                z <= zcut,
                curves[i],
                0.0,
            )

            fills[panel][i] = replace_fill(
                axes[panel],
                fills[panel][i],
                z,
                y,
                colors[i],
            )

            lines[panel][i].set_data(z, y)

            if panel > 0:
                reference_y = np.where(
                    z <= zcut,
                    reference_curves[panel][i],
                    0.0,
                )

                reference_lines[panel][i].set_data(
                    z,
                    reference_y,
                )

    annotations[0].set_text(
        "Equal-width $\\chi$ bins over\n"
        rf"$z \in [{bin_range[0]:.1f},\, {bin_range[1]:.1f}]$"
    )

    artists = [line for panel_lines in lines for line in panel_lines]

    reference_artists = [line for panel_lines in reference_lines for line in panel_lines]

    return (*artists, *reference_artists, *annotations)


def init():
    """Initialize the animation at the minimum redshift."""
    return draw_frame(reveal_vals[0])


def update(i):
    """Update the animation at frame i."""
    return draw_frame(reveal_vals[i])


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=len(reveal_vals),
    init_func=init,
    interval=120,
    blit=False,
)

anim.save(
    OUTFILE,
    writer=PillowWriter(fps=10),
    dpi=300,
)

plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
print(f"Equal-chi redshift edges: {z_edges}")
print(f"Equal-z redshift edges: {equal_z_edges}")
print(f"Comoving-distance edges [Mpc]: {chi_edges}")
