from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, to_rgba

from binny import NZTomography

# Animation controls
FPS = 9
PAUSE_FRAMES = 4
TRANSITION_FRAMES = 6
FIGSIZE = (10.8, 7.8)
USE_CROSSFADE = True


def blend_values(a, b, t):
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def blend_bin_dict(bin_dict_a, bin_dict_b, t):
    keys = sorted(bin_dict_a.keys())
    return {k: blend_values(bin_dict_a[k], bin_dict_b[k], t) for k in keys}


def blend_matrix_dict(mat_a, mat_b, t):
    row_keys = sorted(mat_a.keys())
    col_keys = sorted(mat_a[row_keys[0]].keys())
    return {
        i: {j: (1.0 - t) * float(mat_a[i][j]) + t * float(mat_b[i][j]) for j in col_keys}
        for i in row_keys
    }


def nested_dict_to_matrix(nested_dict):
    keys = sorted(nested_dict.keys())
    matrix = np.array(
        [[nested_dict[row_key][col_key] for col_key in keys] for row_key in keys],
        dtype=float,
    )
    return keys, matrix


def plot_bins(ax, z, bin_dict, colors, title):
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
    ax.set_title(title)


def make_transparent_cmap(name="viridis", alpha=0.78):
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.05, 0.95, 256))
    colors[:, -1] = alpha
    return ListedColormap(colors)


def matrix_max(nested_dict):
    keys = sorted(nested_dict.keys())
    return max(float(nested_dict[i][j]) for i in keys for j in keys)


def plot_matrix(
    ax,
    matrix_dict,
    title,
    xlabel,
    ylabel,
    fmt="{:.1f}",
    cmap=None,
    vmin=None,
    vmax=None,
):
    ax.cla()

    keys, matrix = nested_dict_to_matrix(matrix_dict)
    n_rows, n_cols = matrix.shape

    ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([f"{key + 1}" for key in keys])
    ax.set_yticklabels([f"{key + 1}" for key in keys])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="k",
            )


def plot_leakage_composition(ax, leakage_dict, colors, title):
    ax.cla()

    keys = sorted(leakage_dict.keys())
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys), dtype=float)

    fill_colors = [to_rgba(color, 0.65) for color in colors]

    for fill_color, target_key in zip(fill_colors, keys, strict=True):
        values = np.array(
            [float(leakage_dict[source_key][target_key]) for source_key in keys],
            dtype=float,
        )
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=fill_color,
            edgecolor="k",
            linewidth=1.8,
            label=f"Nominal bin {target_key + 1}",
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"{key + 1}" for key in keys])
    ax.set_xlabel("Source bin")
    ax.set_ylabel("Percent of source-bin mass")
    ax.set_title(title)
    ax.set_ylim(0.0, 100.0)
    ax.legend(frameon=True, loc="center left")


# Output path
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_bin_diagnostics.gif"

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
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": common_uncertainties,
    "normalize_bins": True,
}

equidistant_spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equidistant",
        "n_bins": 4,
        "range": (0.2, 1.2),
    },
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

bin_edges_eqpop = result_eqpop.tomo_meta["bins"]["bin_edges"]
bin_edges_eqdist = result_eqdist.tomo_meta["bins"]["bin_edges"]

stats_eqpop = tomo_eqpop.cross_bin_stats(
    leakage={"bin_edges": bin_edges_eqpop, "unit": "percent", "decimal_places": 3},
)

stats_eqdist = tomo_eqdist.cross_bin_stats(
    leakage={"bin_edges": bin_edges_eqdist, "unit": "percent", "decimal_places": 3},
)

states = [
    {
        "name": "Equipopulated",
        "bins": result_eqpop.bins,
        "leakage": stats_eqpop["leakage"],
    },
    {
        "name": "Equidistant",
        "bins": result_eqdist.bins,
        "leakage": stats_eqdist["leakage"],
    },
]

leakage_vmax = max(matrix_max(state["leakage"]) for state in states)

bin_colors = cmr.take_cmap_colors(
    "viridis",
    4,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

cmap_transparent = make_transparent_cmap("viridis", alpha=0.78)

# Figure layout
fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.2])

ax_bins = fig.add_subplot(gs[0, :])
ax_comp = fig.add_subplot(gs[1, 0])
ax_leakage = fig.add_subplot(gs[1, 1])

fig.suptitle("Tomographic bin diagnostics", fontsize=16)

timeline = []

timeline.extend([("hold", 0, 0.0)] * PAUSE_FRAMES)

if USE_CROSSFADE:
    for i in range(1, TRANSITION_FRAMES + 1):
        t = i / (TRANSITION_FRAMES + 1)
        timeline.append(("transition", 0, t))
else:
    timeline.append(("hold", 1, 0.0))

timeline.extend([("hold", 1, 0.0)] * PAUSE_FRAMES)

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
        leakage = state["leakage"]
        title = f"{state['name']} bins"
        comp_title = f"{state['name']} leakage composition"
        leakage_title = f"{state['name']} leakage matrix"

    else:
        state_a = states[idx]
        state_b = states[(idx + 1) % len(states)]

        bins = blend_bin_dict(state_a["bins"], state_b["bins"], t)
        leakage = blend_matrix_dict(state_a["leakage"], state_b["leakage"], t)
        title = "Comparing diagnostics"
        comp_title = "Leakage composition"
        leakage_title = "Leakage matrix"

    plot_bins(
        ax_bins,
        z,
        bins,
        colors=bin_colors,
        title=title,
    )

    plot_leakage_composition(
        ax_comp,
        leakage,
        colors=bin_colors,
        title=comp_title,
    )

    plot_matrix(
        ax_leakage,
        leakage,
        title=leakage_title,
        xlabel="Nominal bin",
        ylabel="Source bin",
        fmt="{:.1f}",
        cmap=cmap_transparent,
        vmin=0.0,
        vmax=leakage_vmax,
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
