from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba

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

# Animation controls
FPS = 8
PAUSE_FRAMES = 6
TRANSITION_FRAMES = 7
FIGSIZE = (12.0, 9.8)

# If True, smoothly interpolate between consecutive magnitude limits.
USE_CROSSFADE = True


def blend_values(a, b, t):
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def linear_interp(x0, y0, x1, y1, x):
    if np.isclose(x1, x0):
        return float(y0)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def eval_z0_fit(z0_fit, maglim):
    if z0_fit["law"] == "linear":
        return z0_fit["a"] * maglim + z0_fit["b"]
    if z0_fit["law"] == "poly2":
        return z0_fit["c2"] * maglim**2 + z0_fit["c1"] * maglim + z0_fit["c0"]
    raise ValueError(f"Unknown z0 law: {z0_fit['law']}")


def eval_ngal_fit(ngal_fit, maglim):
    if ngal_fit["law"] == "linear":
        return ngal_fit["p"] * maglim + ngal_fit["q"]
    if ngal_fit["law"] == "loglinear":
        return 10.0 ** (ngal_fit["s"] * maglim + ngal_fit["t"])
    raise ValueError(f"Unknown ngal law: {ngal_fit['law']}")


def hist_density(zvals, bins, hist_range):
    hist, edges = np.histogram(zvals, bins=bins, range=hist_range, density=True)
    return hist, edges


def hist_polygon_xy(hist, edges):
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(hist, 2)
    return x, y


def plot_hist_and_fit(
    ax,
    edges,
    hist,
    z_grid,
    nz_fit,
    hist_color,
    fit_color,
):
    ax.cla()

    ax.set_title("Calibrating a parent Smail $n(z)$ from mock galaxies")

    widths = np.diff(edges)
    ax.bar(
        edges[:-1],
        hist,
        width=widths,
        align="edge",
        color=hist_color,
        edgecolor="k",
        linewidth=2.2,
        label="Mock sample",
        zorder=5,
    )

    ax.fill_between(
        z_grid,
        0.0,
        nz_fit,
        color=fit_color,
        alpha=0.65,
        edgecolor="k",
        linewidth=2.4,
        label="Fitted Smail model",
        zorder=20,
    )

    ax.set_xlim(z_grid.min(), z_grid.max())
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n(z)$")
    ax.legend(frameon=False, loc="upper right")


def plot_relations(
    ax_z0,
    ax_ngal,
    mfit,
    z0_curve,
    ngal_curve,
    z0_points_maglim,
    z0_points_vals,
    ngal_points_maglim,
    ngal_points_vals,
    current_maglim,
    current_z0,
    current_ngal,
    color_z0,
    color_ngal,
):
    ax_z0.cla()
    ax_ngal.cla()

    fill_z0 = to_rgba(color_z0, 0.65)
    fill_ngal = to_rgba(color_ngal, 0.65)

    ax_z0.plot(mfit, z0_curve, lw=3.0, color=color_z0, alpha=0.9, zorder=10)
    ax_z0.scatter(
        z0_points_maglim,
        z0_points_vals,
        s=120,
        facecolor=fill_z0,
        edgecolors="k",
        linewidth=1.8,
        zorder=20,
    )
    ax_z0.scatter(
        [current_maglim],
        [current_z0],
        s=220,
        facecolor="white",
        edgecolors="k",
        linewidth=2.2,
        zorder=30,
    )
    ax_z0.scatter(
        [current_maglim],
        [current_z0],
        s=90,
        facecolor=fill_z0,
        edgecolors="none",
        zorder=31,
    )
    ax_z0.axvline(current_maglim, color="k", lw=1.8, ls="--", alpha=0.8, zorder=1)
    ax_z0.set_xlabel(r"Limiting magnitude $m_{\rm lim}$")
    ax_z0.set_ylabel(r"Fitted $z_0$")
    ax_z0.yaxis.set_label_coords(-0.14, 0.5)
    ax_z0.set_title(r"Calibrated $z_0(m_{\rm lim})$")

    ax_ngal.plot(mfit, ngal_curve, lw=3.0, color=color_ngal, alpha=0.9, zorder=10)
    ax_ngal.scatter(
        ngal_points_maglim,
        ngal_points_vals,
        s=120,
        facecolor=fill_ngal,
        edgecolors="k",
        linewidth=1.8,
        zorder=20,
    )
    ax_ngal.scatter(
        [current_maglim],
        [current_ngal],
        s=220,
        facecolor="white",
        edgecolors="k",
        linewidth=2.2,
        zorder=30,
    )
    ax_ngal.scatter(
        [current_maglim],
        [current_ngal],
        s=90,
        facecolor=fill_ngal,
        edgecolors="none",
        zorder=31,
    )
    ax_ngal.axvline(current_maglim, color="k", lw=1.8, ls="--", alpha=0.8, zorder=1)
    ax_ngal.set_xlabel(r"Limiting magnitude $m_{\rm lim}$")
    ax_ngal.set_ylabel(r"$n_{\rm gal}$ [arcmin$^{-2}$]")
    ax_ngal.set_title(r"Calibrated $n_{\rm gal}(m_{\rm lim})$")


def plot_text_panel(
    ax,
    maglim,
    alpha,
    beta,
    z0,
    ngal,
    n_selected,
    frac_selected,
    n_total,
):
    ax.cla()
    ax.axis("off")

    n_rejected = n_total - n_selected

    left_lines = [
        rf"$m_{{\rm lim}} = {maglim:.2f}$",
        rf"$\alpha = {alpha:.3f}$",
        rf"$\beta = {beta:.3f}$",
        rf"$z_0 = {z0:.3f}$",
    ]

    right_lines = [
        rf"$n_{{\rm gal}} = {ngal:.3f}\ \mathrm{{arcmin}}^{{-2}}$",
        f"Selected: {n_selected:,}",
        f"Rejected: {n_rejected:,}",
        f"Fraction kept: {100.0 * frac_selected:.1f}%",
    ]

    text_left = "\n".join(left_lines)
    text_right = "\n".join(right_lines)

    ax.text(
        0.06,
        0.88,
        text_left,
        va="top",
        ha="left",
        linespacing=1.45,
        family="monospace",
        transform=ax.transAxes,
    )

    ax.text(
        0.54,
        0.88,
        text_right,
        va="top",
        ha="left",
        linespacing=1.45,
        family="monospace",
        transform=ax.transAxes,
    )


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "smail_from_mock_calibration.gif"


# Synthetic mock catalog
rng = np.random.default_rng(42)
n_gal = 500_000

z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

result = NZTomography.calibrate_smail_from_mock(
    z_true=z_true,
    mag=mag,
    maglims=maglims,
    area_deg2=1000.0,
    infer_alpha_beta_from="deep_cut",
    alpha_beta_maglim=24.5,
    z_max=3.0,
)

alpha = result["alpha_beta_fit"]["params"]["alpha"]
beta = result["alpha_beta_fit"]["params"]["beta"]

z0_points = result["z0_of_maglim"]["points"]
z0_fit = result["z0_of_maglim"]["fit"]

ngal_points = result["ngal_of_maglim"]["points"]
ngal_fit = result["ngal_of_maglim"]["fit"]

z_grid = np.linspace(0.0, 3.0, 600)
mfit = np.linspace(maglims.min(), maglims.max(), 300)

z0_curve = np.array([eval_z0_fit(z0_fit, m) for m in mfit], dtype=float)
ngal_curve = np.array([eval_ngal_fit(ngal_fit, m) for m in mfit], dtype=float)

# Precompute per-maglim states
states = []
for maglim in maglims:
    sel = mag <= maglim
    z_sel = z_true[sel]

    z0 = eval_z0_fit(z0_fit, maglim)
    ngal = eval_ngal_fit(ngal_fit, maglim)

    nz_fit = NZTomography.nz_model(
        "smail",
        z_grid,
        z0=z0,
        alpha=alpha,
        beta=beta,
        normalize=True,
    )

    hist, edges = hist_density(
        z_sel,
        bins=22,
        hist_range=(0.0, 3.0),
    )

    states.append(
        {
            "maglim": float(maglim),
            "hist": hist,
            "edges": edges,
            "nz_fit": np.asarray(nz_fit, dtype=float),
            "z0": float(z0),
            "ngal": float(ngal),
            "n_selected": int(sel.sum()),
            "frac_selected": float(sel.sum() / mag.size),
        }
    )

# Colors
colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0.05, 0.95),
    return_fmt="hex",
)
c_hist = to_rgba(colors[1], 0.60)
c_fit = colors[3]
c_z0 = colors[1]
c_ngal = colors[3]

# Figure layout
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[1.55, 1.25, 0.72],
    hspace=0.5,
    wspace=0.35,
)

ax_main = fig.add_subplot(gs[0, :])
ax_z0 = fig.add_subplot(gs[1, 0])
ax_ngal = fig.add_subplot(gs[1, 1])
ax_text = fig.add_subplot(gs[2, :])

# Timeline
timeline = []

for idx in range(len(states)):
    timeline.extend([("hold", idx, 0.0)] * PAUSE_FRAMES)

    if idx < len(states) - 1:
        if USE_CROSSFADE:
            for i in range(1, TRANSITION_FRAMES + 1):
                t = i / (TRANSITION_FRAMES + 1)
                timeline.append(("transition", idx, t))
        else:
            timeline.append(("hold", idx + 1, 0.0))

# pause at final state a bit more
timeline.extend([("hold", len(states) - 1, 0.0)] * (PAUSE_FRAMES + 2))


def blended_state(idx, t):
    a = states[idx]
    b = states[idx + 1]

    return {
        "maglim": (1.0 - t) * a["maglim"] + t * b["maglim"],
        "hist": blend_values(a["hist"], b["hist"], t),
        "edges": a["edges"],
        "nz_fit": blend_values(a["nz_fit"], b["nz_fit"], t),
        "z0": (1.0 - t) * a["z0"] + t * b["z0"],
        "ngal": (1.0 - t) * a["ngal"] + t * b["ngal"],
        "n_selected": int(round((1.0 - t) * a["n_selected"] + t * b["n_selected"])),
        "frac_selected": (1.0 - t) * a["frac_selected"] + t * b["frac_selected"],
    }


def update(frame):
    mode, idx, t = timeline[frame]

    if mode == "hold":
        state = states[idx]
    else:
        state = blended_state(idx, t)

    plot_hist_and_fit(
        ax_main,
        edges=state["edges"],
        hist=state["hist"],
        z_grid=z_grid,
        nz_fit=state["nz_fit"],
        hist_color=c_hist,
        fit_color=c_fit,
    )

    plot_relations(
        ax_z0=ax_z0,
        ax_ngal=ax_ngal,
        mfit=mfit,
        z0_curve=z0_curve,
        ngal_curve=ngal_curve,
        z0_points_maglim=np.asarray(z0_points["maglim"], dtype=float),
        z0_points_vals=np.asarray(z0_points["z0"], dtype=float),
        ngal_points_maglim=np.asarray(ngal_points["maglim"], dtype=float),
        ngal_points_vals=np.asarray(ngal_points["ngal_arcmin2"], dtype=float),
        current_maglim=state["maglim"],
        current_z0=state["z0"],
        current_ngal=state["ngal"],
        color_z0=c_z0,
        color_ngal=c_ngal,
    )

    plot_text_panel(
        ax_text,
        maglim=state["maglim"],
        alpha=alpha,
        beta=beta,
        z0=state["z0"],
        ngal=state["ngal"],
        n_selected=state["n_selected"],
        frac_selected=state["frac_selected"],
        n_total=mag.size,
    )

    return []


anim = FuncAnimation(
    fig,
    update,
    frames=len(timeline),
    interval=1000 / FPS,
    blit=False,
    repeat=True,
)

anim.save(
    OUTFILE,
    writer=PillowWriter(fps=FPS),
)

plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
