from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from binny import NZTomography

DEFAULT_FONTSIZE = 19
plt.rcParams.update(
    {
        "font.size": DEFAULT_FONTSIZE,
        "axes.titlesize": DEFAULT_FONTSIZE,
        "axes.labelsize": DEFAULT_FONTSIZE,
        "xtick.labelsize": DEFAULT_FONTSIZE,
        "ytick.labelsize": DEFAULT_FONTSIZE,
        "legend.fontsize": DEFAULT_FONTSIZE,
        "figure.titlesize": DEFAULT_FONTSIZE,
    }
)


def smail(z, z0, alpha, beta):
    return NZTomography.nz_model(
        "smail",
        z,
        z0=z0,
        alpha=alpha,
        beta=beta,
        normalize=True,
    )


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "smail_parameter_sweep.gif"

z = np.linspace(0.0, 3.0, 500)

alpha0 = 2.0
beta0 = 1.2
z00 = 0.35

# Use same number of frames for all three sweeps
n_frames = 60
alpha_vals = np.linspace(0.5, 4.0, n_frames)
beta_vals = np.linspace(0.6, 2.5, n_frames)
z0_vals = np.linspace(0.15, 0.8, n_frames)

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)
color_alpha, color_beta, color_z0 = colors

# Precompute ymax for stable axes across frames
all_curves = []
for a in alpha_vals:
    all_curves.append(smail(z, z0=z00, alpha=a, beta=beta0))
for b in beta_vals:
    all_curves.append(smail(z, z0=z00, alpha=alpha0, beta=b))
for zz in z0_vals:
    all_curves.append(smail(z, z0=zz, alpha=alpha0, beta=beta0))

ymax = 1.08 * max(np.max(curve) for curve in all_curves)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)

titles = [
    r"Varying $\alpha$",
    r"Varying $\beta$",
    r"Varying $z_0$",
]


for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, ymax)

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=2.0,
        length=6,
    )
    ax.grid(False)

axes[0].set_ylabel(r"Normalized $n(z)$")

# Baseline text under titles
axes[0].text(
    0.97,
    0.96,
    rf"$z_0={z00:.2f}$, $\beta={beta0:.2f}$",
    transform=axes[0].transAxes,
    ha="right",
    va="top",
)
axes[1].text(
    0.97,
    0.96,
    rf"$z_0={z00:.2f}$, $\alpha={alpha0:.2f}$",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
)
axes[2].text(
    0.97,
    0.96,
    rf"$\alpha={alpha0:.2f}$, $\beta={beta0:.2f}$",
    transform=axes[2].transAxes,
    ha="right",
    va="top",
)

text_alpha = axes[0].text(
    0.97,
    0.84,
    "",
    transform=axes[0].transAxes,
    ha="right",
    va="top",
)

text_beta = axes[1].text(
    0.97,
    0.84,
    "",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
)

text_z0 = axes[2].text(
    0.97,
    0.84,
    "",
    transform=axes[2].transAxes,
    ha="right",
    va="top",
)

# Black baselines
for ax in axes:
    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=100)

# Filled curves + black outlines
fill_alpha = axes[0].fill_between(
    z, 0.0, np.zeros_like(z), color=color_alpha, alpha=0.6, linewidth=0.0, zorder=10
)
fill_beta = axes[1].fill_between(
    z, 0.0, np.zeros_like(z), color=color_beta, alpha=0.6, linewidth=0.0, zorder=10
)
fill_z0 = axes[2].fill_between(
    z, 0.0, np.zeros_like(z), color=color_z0, alpha=0.6, linewidth=0.0, zorder=10
)

(line_alpha,) = axes[0].plot([], [], color="k", linewidth=2.0, zorder=20)
(line_beta,) = axes[1].plot([], [], color="k", linewidth=2.0, zorder=20)
(line_z0,) = axes[2].plot([], [], color="k", linewidth=2.0, zorder=20)


def _replace_fill(ax, old_fill, x, y, color):
    old_fill.remove()
    new_fill = ax.fill_between(
        x,
        0.0,
        y,
        color=color,
        alpha=0.6,
        linewidth=0.0,
        zorder=10,
    )
    return new_fill


def init():
    global fill_alpha, fill_beta, fill_z0

    y_alpha = smail(z, z0=z00, alpha=alpha_vals[0], beta=beta0)
    y_beta = smail(z, z0=z00, alpha=alpha0, beta=beta_vals[0])
    y_z0 = smail(z, z0=z0_vals[0], alpha=alpha0, beta=beta0)

    fill_alpha = _replace_fill(axes[0], fill_alpha, z, y_alpha, color_alpha)
    fill_beta = _replace_fill(axes[1], fill_beta, z, y_beta, color_beta)
    fill_z0 = _replace_fill(axes[2], fill_z0, z, y_z0, color_z0)

    line_alpha.set_data(z, y_alpha)
    line_beta.set_data(z, y_beta)
    line_z0.set_data(z, y_z0)

    text_alpha.set_text(rf"$\alpha = {alpha_vals[0]:.2f}$")
    text_beta.set_text(rf"$\beta = {beta_vals[0]:.2f}$")
    text_z0.set_text(rf"$z_0 = {z0_vals[0]:.2f}$")

    return line_alpha, line_beta, line_z0, text_alpha, text_beta, text_z0


def update(i):
    global fill_alpha, fill_beta, fill_z0

    a = alpha_vals[i]
    b = beta_vals[i]
    zz = z0_vals[i]

    y_alpha = smail(z, z0=z00, alpha=a, beta=beta0)
    y_beta = smail(z, z0=z00, alpha=alpha0, beta=b)
    y_z0 = smail(z, z0=zz, alpha=alpha0, beta=beta0)

    fill_alpha = _replace_fill(axes[0], fill_alpha, z, y_alpha, color_alpha)
    fill_beta = _replace_fill(axes[1], fill_beta, z, y_beta, color_beta)
    fill_z0 = _replace_fill(axes[2], fill_z0, z, y_z0, color_z0)

    line_alpha.set_data(z, y_alpha)
    line_beta.set_data(z, y_beta)
    line_z0.set_data(z, y_z0)

    text_alpha.set_text(rf"$\alpha = {a:.2f}$")
    text_beta.set_text(rf"$\beta = {b:.2f}$")
    text_z0.set_text(rf"$z_0 = {zz:.2f}$")

    return line_alpha, line_beta, line_z0, text_alpha, text_beta, text_z0


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=120,
    blit=False,
)

anim.save(
    OUTFILE,
    writer=PillowWriter(fps=10),
)

plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
