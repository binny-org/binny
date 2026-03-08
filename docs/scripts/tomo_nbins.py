from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from binny import NZTomography


def replace_fill(ax, old_fill, x, y, color, alpha=0.6):
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
    keys = sorted(bin_dict.keys())
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def state_colors(n_bins, cmap_name="viridis"):
    if n_bins == 1:
        return cmr.take_cmap_colors(
            cmap_name,
            1,
            cmap_range=(0.5, 0.5),
            return_fmt="hex",
        )
    return cmr.take_cmap_colors(
        cmap_name,
        n_bins,
        cmap_range=(0.0, 1.0),
        return_fmt="hex",
    )


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "tomo_increasing_nbins.gif"

tomo = NZTomography()

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

bin_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
max_bins = max(bin_counts)

states = []
for n_bins in bin_counts:
    spec = {
        "kind": "photoz",
        "bins": {
            "scheme": "equipopulated",
            "n_bins": n_bins,
        },
        "uncertainties": {
            "scatter_scale": 0.05,
        },
        "normalize_bins": True,
    }
    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)
    n_pairs = n_bins * (n_bins + 1) // 2
    states.append((n_bins, curves, n_pairs))

ymax = 1.08 * max(np.max(curve) for _, curves, _ in states for curve in curves)

fig, ax = plt.subplots(figsize=(8.2, 5.0))

title_fs = 19
label_fs = 17
tick_fs = 15
annot_fs = 17
lw = 2.0

ax.set_title("Increasing number of tomographic bins", fontsize=title_fs)
ax.set_xlim(0.0, 2.0)
ax.set_ylim(0.0, ymax)
ax.set_xlabel("Redshift $z$", fontsize=label_fs)
ax.set_ylabel(r"Normalized $n_i(z)$", fontsize=label_fs)

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
ax.plot(z, np.zeros_like(z), color="k", linewidth=lw, zorder=100)

fills = []
lines = []

for _i in range(max_bins):
    fill = ax.fill_between(
        z,
        0.0,
        np.zeros_like(z),
        color="k",
        alpha=0.0,
        linewidth=0.0,
        zorder=10,
    )
    (line,) = ax.plot([], [], color="k", linewidth=lw, zorder=20)
    fills.append(fill)
    lines.append(line)

text_bins = ax.text(0.8, 0.94, "", transform=ax.transAxes, fontsize=annot_fs, ha="left", va="top")
text_pairs = ax.text(0.6, 0.86, "", transform=ax.transAxes, fontsize=annot_fs, ha="left", va="top")

growth_frames = 6
pause_frames = 15
frames_per_state = growth_frames + pause_frames
n_frames = frames_per_state * len(states)


def init():
    global fills

    for i in range(max_bins):
        fills[i] = replace_fill(ax, fills[i], z, np.zeros_like(z), "white", alpha=0.0)
        lines[i].set_data([], [])

    text_bins.set_text("")
    text_pairs.set_text("")
    return *lines, text_bins, text_pairs


def update(frame):
    global fills

    state_idx = min(frame // frames_per_state, len(states) - 1)
    local_frame = frame % frames_per_state

    n_bins, curves, n_pairs = states[state_idx]
    current_colors = state_colors(n_bins)

    if local_frame < growth_frames:
        frac = (local_frame + 1) / growth_frames
    else:
        frac = 1.0

    for i in range(max_bins):
        if i < n_bins:
            y = frac * curves[i]
            fills[i] = replace_fill(ax, fills[i], z, y, current_colors[i])
            lines[i].set_data(z, y)
        else:
            fills[i] = replace_fill(ax, fills[i], z, np.zeros_like(z), "white", alpha=0.0)
            lines[i].set_data([], [])

    text_bins.set_text(rf"$N_{{\mathrm{{bin}}}} = {n_bins}$")
    text_pairs.set_text(rf"Auto + cross pairs: {n_pairs}")

    return *lines, text_bins, text_pairs


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=120,
    blit=False,
)

anim.save(OUTFILE, writer=PillowWriter(fps=8))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")
