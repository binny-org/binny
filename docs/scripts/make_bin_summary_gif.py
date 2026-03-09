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
FINAL_PAUSE_FRAMES = 12
TRANSITION_FRAMES = 8
FIGSIZE = (10.8, 7.8)

# If True, blend between the two states.
USE_CROSSFADE = True


def blend_values(a, b, t):
    return (1.0 - t) * np.asarray(a) + t * np.asarray(b)


def blend_bin_dict(bin_dict_a, bin_dict_b, t):
    keys = sorted(bin_dict_a.keys())
    return {k: blend_values(bin_dict_a[k], bin_dict_b[k], t) for k in keys}


def blend_center_dict(center_dict_a, center_dict_b, t):
    methods = center_dict_a.keys()
    out = {}
    for m in methods:
        keys = sorted(center_dict_a[m].keys())
        out[m] = {k: (1.0 - t) * center_dict_a[m][k] + t * center_dict_b[m][k] for k in keys}
    return out


def blend_fraction_dict(frac_a, frac_b, t):
    keys = sorted(frac_a.keys())
    return {k: (1.0 - t) * frac_a[k] + t * frac_b[k] for k in keys}


def add_scheme_annotation(ax, text):
    ax.text(
        0.98,
        0.94,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        zorder=5000,
    )


def plot_bins(ax, z, bin_dict, colors, annotation):
    ax.cla()

    keys = sorted(bin_dict.keys())
    for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
        curve = np.asarray(bin_dict[key], dtype=float)
        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=0.65,
            linewidth=0.0,
            zorder=10 + i,
        )
        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)
    ax.set_xlim(z.min(), z.max())
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")
    add_scheme_annotation(ax, annotation)


def plot_centers(ax, centers):
    ax.cla()

    keys = sorted(centers["mean"].keys())
    x = np.arange(len(keys))

    method_order = ["mean", "median", "mode"]
    marker_map = {
        "mean": "o",
        "median": "s",
        "mode": "^",
    }
    label_map = {
        "mean": "Mean",
        "median": "Median",
        "mode": "Mode",
    }
    offset_map = {
        "mean": -0.18,
        "median": 0.0,
        "mode": 0.18,
    }

    summary_colors = cmr.take_cmap_colors(
        "viridis",
        3,
        cmap_range=(0.15, 0.9),
        return_fmt="hex",
    )

    for method, c in zip(method_order, summary_colors, strict=True):
        ax.scatter(
            x + offset_map[method],
            [centers[method][key] for key in keys],
            marker=marker_map[method],
            s=150,
            facecolors=to_rgba(c, 0.7),
            edgecolors="k",
            linewidths=1.4,
            label=label_map[method],
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k + 1}" for k in keys])
    ax.set_xlabel("Tomographic bin")
    ax.set_ylabel("Redshift $z$")
    ax.legend(frameon=True)


def plot_fractions(ax, fractions, colors):
    ax.cla()

    keys = sorted(fractions.keys())
    x = np.arange(len(keys))
    vals = [fractions[k] for k in keys]

    ax.bar(
        x,
        vals,
        width=0.65,
        color=[to_rgba(c, 0.65) for c in colors],
        edgecolor="k",
        linewidth=1.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k + 1}" for k in keys])
    ax.set_xlabel("Tomographic bin")
    ax.set_ylabel("Galaxy fraction")


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_bin_summaries.gif"

# Shared setup
z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

common_uncertainties = {
    "scatter_scale": [0.010, 0.012, 0.015, 0.018],
    "mean_offset": 0.0,
    "outlier_frac": [0.02, 0.05, 0.15, 0.26],
    "outlier_scatter_scale": [0.008, 0.010, 0.012, 0.015],
    "outlier_mean_offset": [0.35, 0.40, 0.45, 0.50],
}

equipopulated_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equipopulated", "n_bins": 4},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_spec = {
    "kind": "photoz",
    "bins": {"scheme": "equidistant", "n_bins": 4, "range": (0.2, 1.2)},
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

tomo_eqpop = NZTomography()
result_eqpop = tomo_eqpop.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equipopulated_spec,
    include_tomo_metadata=True,
)

tomo_eqdist = NZTomography()
result_eqdist = tomo_eqdist.build_bins(
    z=z,
    nz=nz,
    tomo_spec=equidistant_spec,
    include_tomo_metadata=True,
)

states = [
    {
        "name": "Equipopulated",
        "annotation": "Scheme: equipopulated",
        "bins": result_eqpop.bins,
        "centers": {
            m: tomo_eqpop.shape_stats(center_method=m, decimal_places=3)["centers"]
            for m in ["mean", "median", "mode"]
        },
        "fractions": tomo_eqpop.population_stats(decimal_places=4)["fractions"],
    },
    {
        "name": "Equidistant",
        "annotation": "Scheme: equidistant",
        "bins": result_eqdist.bins,
        "centers": {
            m: tomo_eqdist.shape_stats(center_method=m, decimal_places=3)["centers"]
            for m in ["mean", "median", "mode"]
        },
        "fractions": tomo_eqdist.population_stats(decimal_places=4)["fractions"],
    },
]

bin_colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

# Figure layout
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.2, 1.15],
    hspace=0.35,
    wspace=0.28,
)

ax_bins = fig.add_subplot(gs[0, :])
ax_centers = fig.add_subplot(gs[1, 0])
ax_fractions = fig.add_subplot(gs[1, 1])

fig.suptitle("Representative tomographic bins")

# Build animation timeline
timeline = []

# hold state 0
timeline.extend([("hold", 0, 0.0)] * PAUSE_FRAMES)

# transition 0 -> 1
if USE_CROSSFADE:
    for i in range(1, TRANSITION_FRAMES + 1):
        t = i / (TRANSITION_FRAMES + 1)
        timeline.append(("transition", 0, t))
else:
    timeline.append(("hold", 1, 0.0))

# hold state 1
timeline.extend([("hold", 1, 0.0)] * FINAL_PAUSE_FRAMES)

# transition 1 -> 0
if USE_CROSSFADE:
    for i in range(1, TRANSITION_FRAMES + 1):
        t = i / (TRANSITION_FRAMES + 1)
        timeline.append(("transition", 1, t))
else:
    timeline.append(("hold", 0, 0.0))


def update(frame):
    mode, idx, t = timeline[frame]

    if mode == "hold":
        state = states[idx]
        bins = state["bins"]
        centers = state["centers"]
        fractions = state["fractions"]
        annotation = state["annotation"]

    else:
        state_a = states[idx]
        state_b = states[(idx + 1) % len(states)]

        bins = blend_bin_dict(state_a["bins"], state_b["bins"], t)
        centers = blend_center_dict(state_a["centers"], state_b["centers"], t)
        fractions = blend_fraction_dict(state_a["fractions"], state_b["fractions"], t)

        if idx == 0:
            annotation = "Scheme: equipopulated → equidistant"
        else:
            annotation = "Scheme: equidistant → equipopulated"

    plot_bins(
        ax_bins,
        z,
        bins,
        colors=bin_colors,
        annotation=annotation,
    )

    plot_centers(
        ax_centers,
        centers,
    )

    plot_fractions(
        ax_fractions,
        fractions,
        colors=bin_colors,
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
