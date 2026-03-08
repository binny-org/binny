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


def shifted_smail(z, z0, alpha, beta, z_shift):
    return NZTomography.nz_model(
        "shifted_smail",
        z,
        z0=z0,
        alpha=alpha,
        beta=beta,
        z_shift=z_shift,
        normalize=True,
    )


def gaussian_mixture(z, mus, sigmas, weights):
    return NZTomography.nz_model(
        "gaussian_mixture",
        z,
        mus=np.asarray(mus, dtype=float),
        sigmas=np.asarray(sigmas, dtype=float),
        weights=np.asarray(weights, dtype=float),
        normalize=True,
    )


def tophat(z, zmin, zmax):
    return NZTomography.nz_model(
        "tophat",
        z,
        zmin=zmin,
        zmax=zmax,
        normalize=True,
    )


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "parent_nz_model_sweep.gif"

z = np.linspace(0.0, 3.0, 500)

alpha0 = 2.0
beta0 = 1.2
z00 = 0.35

n_frames = 60
alpha_vals = np.linspace(0.5, 4.0, n_frames)
beta_vals = np.linspace(0.6, 2.5, n_frames)
z0_vals = np.linspace(0.15, 0.8, n_frames)

frame_sequence = list(range(n_frames)) + list(range(n_frames - 2, 0, -1))

# Bottom row animation controls
zshift_vals = np.linspace(0.0, 0.55, n_frames)

mu1_vals = np.linspace(0.45, 0.80, n_frames)
mu2_vals = np.linspace(1.05, 1.55, n_frames)
sigma1 = 0.12
sigma2 = 0.18
weights_mix = np.array([0.45, 0.55])

center_vals = np.linspace(0.55, 1.75, n_frames)
width0 = 0.5

colors = cmr.take_cmap_colors(
    "viridis",
    6,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)
(
    color_alpha,
    color_beta,
    color_z0,
    color_shift,
    color_mix,
    color_tophat,
) = colors

all_curves = []

for a in alpha_vals:
    all_curves.append(smail(z, z0=z00, alpha=a, beta=beta0))

for b in beta_vals:
    all_curves.append(smail(z, z0=z00, alpha=alpha0, beta=b))

for zz in z0_vals:
    all_curves.append(smail(z, z0=zz, alpha=alpha0, beta=beta0))

for zshift in zshift_vals:
    all_curves.append(shifted_smail(z, z0=z00, alpha=alpha0, beta=beta0, z_shift=zshift))

for mu1, mu2 in zip(mu1_vals, mu2_vals, strict=True):
    all_curves.append(
        gaussian_mixture(
            z,
            mus=[mu1, mu2],
            sigmas=[sigma1, sigma2],
            weights=weights_mix,
        )
    )

for center in center_vals:
    zmin = max(0.0, center - 0.5 * width0)
    zmax = min(3.0, center + 0.5 * width0)
    all_curves.append(tophat(z, zmin=zmin, zmax=zmax))

ymax = 1.08 * max(np.max(curve) for curve in all_curves)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), sharex=True, sharey=True)
axes = axes.ravel()

fig.text(
    0.5,
    0.98,
    r"Smail parent $n(z)$ parameter variations",
    ha="center",
    va="top",
    fontsize=22,
)

titles = [
    r"Varying $\alpha$",
    r"Varying $\beta$",
    r"Varying $z_0$",
    r"Shifted Smail",
    r"Gaussian mixture",
    r"Top-hat",
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
axes[3].set_ylabel(r"Normalized $n(z)$")

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

axes[3].text(
    0.97,
    0.96,
    rf"$z_0={z00:.2f}$, $\alpha={alpha0:.2f}$, $\beta={beta0:.2f}$",
    transform=axes[3].transAxes,
    ha="right",
    va="top",
)

axes[4].text(
    0.97,
    0.96,
    rf"$\sigma_1={sigma1:.2f}$, $\sigma_2={sigma2:.2f}$",
    transform=axes[4].transAxes,
    ha="right",
    va="top",
)

axes[5].text(
    0.97,
    0.96,
    rf"$\Delta z={width0:.2f}$",
    transform=axes[5].transAxes,
    ha="right",
    va="top",
)

text_alpha = axes[0].text(0.97, 0.84, "", transform=axes[0].transAxes, ha="right", va="top")
text_beta = axes[1].text(0.97, 0.84, "", transform=axes[1].transAxes, ha="right", va="top")
text_z0 = axes[2].text(0.97, 0.84, "", transform=axes[2].transAxes, ha="right", va="top")
text_shift = axes[3].text(0.97, 0.84, "", transform=axes[3].transAxes, ha="right", va="top")
text_mix = axes[4].text(0.97, 0.84, "", transform=axes[4].transAxes, ha="right", va="top")
text_tophat = axes[5].text(0.97, 0.84, "", transform=axes[5].transAxes, ha="right", va="top")

for ax in axes:
    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=100)

fills = []
for ax, color in zip(
    axes,
    [color_alpha, color_beta, color_z0, color_shift, color_mix, color_tophat],
    strict=True,
):
    fills.append(
        ax.fill_between(
            z,
            0.0,
            np.zeros_like(z),
            color=color,
            alpha=0.6,
            linewidth=0.0,
            zorder=10,
        )
    )

lines = []
for ax in axes:
    (line,) = ax.plot([], [], color="k", linewidth=2.0, zorder=20)
    lines.append(line)


def _replace_fill(ax, old_fill, x, y, color):
    old_fill.remove()
    return ax.fill_between(
        x,
        0.0,
        y,
        color=color,
        alpha=0.6,
        linewidth=0.0,
        zorder=10,
    )


def _bottom_row_curves(i):
    zshift = zshift_vals[i]

    mu1 = mu1_vals[i]
    mu2 = mu2_vals[i]
    y_mix = gaussian_mixture(
        z,
        mus=[mu1, mu2],
        sigmas=[sigma1, sigma2],
        weights=weights_mix,
    )

    center = center_vals[i]
    zmin = max(0.0, center - 0.5 * width0)
    zmax = min(3.0, center + 0.5 * width0)
    y_tophat = tophat(z, zmin=zmin, zmax=zmax)

    y_shift = shifted_smail(
        z,
        z0=z00,
        alpha=alpha0,
        beta=beta0,
        z_shift=zshift,
    )

    return y_shift, y_mix, y_tophat, zshift, mu1, mu2, zmin, zmax


def init():
    global fills

    y_alpha = smail(z, z0=z00, alpha=alpha_vals[0], beta=beta0)
    y_beta = smail(z, z0=z00, alpha=alpha0, beta=beta_vals[0])
    y_z0 = smail(z, z0=z0_vals[0], alpha=alpha0, beta=beta0)

    y_shift, y_mix, y_tophat, zshift, mu1, mu2, zmin, zmax = _bottom_row_curves(0)

    y_all = [y_alpha, y_beta, y_z0, y_shift, y_mix, y_tophat]
    colors_all = [color_alpha, color_beta, color_z0, color_shift, color_mix, color_tophat]

    for j in range(6):
        fills[j] = _replace_fill(axes[j], fills[j], z, y_all[j], colors_all[j])
        lines[j].set_data(z, y_all[j])

    text_alpha.set_text(rf"$\alpha = {alpha_vals[0]:.2f}$")
    text_beta.set_text(rf"$\beta = {beta_vals[0]:.2f}$")
    text_z0.set_text(rf"$z_0 = {z0_vals[0]:.2f}$")
    text_shift.set_text(rf"$z_{{\rm shift}} = {zshift:.1f}$")
    text_mix.set_text(rf"$\mu_1 = {mu1:.1f},\ \mu_2 = {mu2:.1f}$")
    text_tophat.set_text(rf"$z_{{\min}} = {zmin:.1f},\ z_{{\max}} = {zmax:.1f}$")

    return (*lines, text_alpha, text_beta, text_z0, text_shift, text_mix, text_tophat)


def update(i):
    global fills

    i = frame_sequence[i]

    a = alpha_vals[i]
    b = beta_vals[i]
    zz = z0_vals[i]

    y_alpha = smail(z, z0=z00, alpha=a, beta=beta0)
    y_beta = smail(z, z0=z00, alpha=alpha0, beta=b)
    y_z0 = smail(z, z0=zz, alpha=alpha0, beta=beta0)

    y_shift, y_mix, y_tophat, zshift, mu1, mu2, zmin, zmax = _bottom_row_curves(i)

    y_all = [y_alpha, y_beta, y_z0, y_shift, y_mix, y_tophat]
    colors_all = [color_alpha, color_beta, color_z0, color_shift, color_mix, color_tophat]

    for j in range(6):
        fills[j] = _replace_fill(axes[j], fills[j], z, y_all[j], colors_all[j])
        lines[j].set_data(z, y_all[j])

    text_alpha.set_text(rf"$\alpha = {a:.2f}$")
    text_beta.set_text(rf"$\beta = {b:.2f}$")
    text_z0.set_text(rf"$z_0 = {zz:.2f}$")
    text_shift.set_text(rf"$z_{{\rm shift}} = {zshift:.1f}$")
    text_mix.set_text(rf"$\mu_1 = {mu1:.1f},\ \mu_2 = {mu2:.1f}$")
    text_tophat.set_text(rf"$z_{{\min}} = {zmin:.1f},\ z_{{\max}} = {zmax:.1f}$")

    return (*lines, text_alpha, text_beta, text_z0, text_shift, text_mix, text_tophat)


plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0, w_pad=1.0)

anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_sequence),
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
