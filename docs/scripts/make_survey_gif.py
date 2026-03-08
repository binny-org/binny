from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import yaml
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

FPS = 8
PAUSE_FRAMES = 6
TRANSITION_FRAMES = 8
FIGSIZE = (8.8, 7.2)
USE_CROSSFADE = True


def blend_values(a, b, t):
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def blend_bin_dict(bin_dict_a, bin_dict_b, t):
    keys_a = set(bin_dict_a.keys())
    keys_b = set(bin_dict_b.keys())
    all_keys = sorted(keys_a | keys_b)

    template = None
    if bin_dict_a:
        template = np.asarray(next(iter(bin_dict_a.values())), dtype=float)
    elif bin_dict_b:
        template = np.asarray(next(iter(bin_dict_b.values())), dtype=float)
    else:
        return {}

    zeros = np.zeros_like(template, dtype=float)

    blended = {}
    for k in all_keys:
        arr_a = np.asarray(bin_dict_a.get(k, zeros), dtype=float)
        arr_b = np.asarray(bin_dict_b.get(k, zeros), dtype=float)
        blended[k] = blend_values(arr_a, arr_b, t)

    return blended


def plot_bins(ax, z, bin_dict, title, label=None, xlim=None):
    ax.cla()

    keys = sorted(bin_dict.keys())
    colors = cmr.take_cmap_colors(
        "viridis",
        len(keys),
        cmap_range=(0.1, 0.9),
        return_fmt="hex",
    )

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

    if xlim is None:
        ax.set_xlim(z.min(), z.max())
    else:
        ax.set_xlim(*xlim)

    ymax = max(
        (np.max(np.asarray(bin_dict[key], dtype=float)) for key in keys),
        default=1.0,
    )
    ax.set_ylim(0.0, 1.08 * ymax)

    if label is not None:
        ax.text(
            0.97,
            0.92,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=DEFAULT_FONTSIZE,
        )

    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")


def build_tomo_spec(entry):
    return {
        "kind": entry["kind"],
        "bins": entry["bins"],
        "uncertainties": entry["uncertainties"],
        "normalize_bins": True,
    }


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "lsst_preset_sweep.gif"

preset_path = (
    HERE.parent.parent / "src" / "binny" / "surveys" / "configs" / "lsst_survey_specs.yaml"
)

with preset_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

z_cfg = config["z_grid"]
z = np.linspace(
    z_cfg["start"],
    z_cfg["stop"],
    z_cfg["n"],
)

entries = config["tomography"]

selected = {
    ("lens", "1"): None,
    ("source", "1"): None,
    ("lens", "10"): None,
    ("source", "10"): None,
}

for entry in entries:
    key = (entry["role"], entry["year"])
    if key in selected:
        selected[key] = entry

results = {}

for key, entry in selected.items():
    nz = NZTomography.nz_model(
        entry["nz"]["model"],
        z,
        normalize=True,
        **entry["nz"]["params"],
    )

    tomo = NZTomography()
    result = tomo.build_bins(
        z=z,
        nz=nz,
        tomo_spec=build_tomo_spec(entry),
        include_tomo_metadata=True,
    )

    results[key] = result

states = [
    {
        "label": "Y1",
        "lens_bins": results[("lens", "1")].bins,
        "source_bins": results[("source", "1")].bins,
    },
    {
        "label": "Y10",
        "lens_bins": results[("lens", "10")].bins,
        "source_bins": results[("source", "10")].bins,
    },
]

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

fig, axes = plt.subplots(
    2,
    1,
    figsize=FIGSIZE,
    constrained_layout=True,
    sharex=False,
)

ax_lens, ax_source = axes


def update(frame):
    mode, idx, t = timeline[frame]

    if mode == "hold":
        state = states[idx]
        lens_bins = state["lens_bins"]
        source_bins = state["source_bins"]
        label = state["label"]

    else:
        state_a = states[idx]
        state_b = states[(idx + 1) % len(states)]

        lens_bins = blend_bin_dict(state_a["lens_bins"], state_b["lens_bins"], t)
        source_bins = blend_bin_dict(state_a["source_bins"], state_b["source_bins"], t)
        label = f"{state_a['label']} \u2192 {state_b['label']}"

    plot_bins(
        ax_lens,
        z,
        lens_bins,
        title="LSST lens sample",
        label=label,
        xlim=(0.0, 1.5),
    )

    plot_bins(
        ax_source,
        z,
        source_bins,
        title="LSST source sample",
        label=label,
        xlim=(0.0, z.max()),
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
