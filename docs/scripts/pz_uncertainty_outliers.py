from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from binny import NZTomography

DEFAULT_FONTSIZE = 17
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

FPS = 10
FIGSIZE = (11.4, 9.0)


def ordered_curves(bin_dict):
    keys = sorted(bin_dict.keys())
    return [np.asarray(bin_dict[k], dtype=float) for k in keys]


def make_pingpong_indices(n):
    forward = list(range(n))
    backward = list(range(n - 2, 0, -1))
    return forward + backward


def fill_vertices(x, y):
    return np.vstack(
        [
            [x[0], 0.0],
            np.column_stack([x, y]),
            [x[-1], 0.0],
        ]
    )


def gaussian_profile(u, center, sigma):
    return np.exp(-0.5 * ((u - center) / sigma) ** 2)


def make_spec(
    n_bins,
    bin_range,
    scatter_scale,
    mean_scale,
    mean_offset,
    outlier_frac,
    outlier_scatter_scale,
    outlier_mean_scale,
    outlier_mean_offset,
):
    return {
        "kind": "photoz",
        "bins": {
            "scheme": "equidistant",
            "n_bins": n_bins,
            "range": bin_range,
        },
        "uncertainties": {
            "scatter_scale": scatter_scale,
            "mean_scale": mean_scale,
            "mean_offset": mean_offset,
            "outlier_frac": outlier_frac,
            "outlier_scatter_scale": outlier_scatter_scale,
            "outlier_mean_scale": outlier_mean_scale,
            "outlier_mean_offset": outlier_mean_offset,
        },
        "normalize_bins": True,
    }


HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "_static" / "animations"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "pz_uncertainty_outliers.gif"

tomo = NZTomography()

z = np.linspace(0.0, 2.0, 180)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

n_bins = 3
bin_range = (0.2, 1.1)

colors = cmr.take_cmap_colors(
    "viridis",
    n_bins,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

baseline_scatter = 0.03
baseline_mean_scale = 1.0
baseline_mean_offset = 0.0

baseline_outlier_frac = 0.0
baseline_outlier_scatter = 0.12
baseline_outlier_mean_scale = 1.0
baseline_outlier_mean_offset = 0.25

n_states = 16

outlier_frac_vals = np.linspace(0.0, 0.60, n_states)
outlier_mean_offset_vals = np.linspace(0.0, 0.80, n_states)
outlier_mean_scale_vals = np.linspace(1.0, 1.85, n_states)
outlier_scatter_vals = np.linspace(0.05, 0.75, n_states)

frame_ids = make_pingpong_indices(n_states)

base_spec = make_spec(
    n_bins=n_bins,
    bin_range=bin_range,
    scatter_scale=baseline_scatter,
    mean_scale=baseline_mean_scale,
    mean_offset=baseline_mean_offset,
    outlier_frac=baseline_outlier_frac,
    outlier_scatter_scale=baseline_outlier_scatter,
    outlier_mean_scale=baseline_outlier_mean_scale,
    outlier_mean_offset=baseline_outlier_mean_offset,
)

base_result = tomo.build_bins(z=z, nz=nz, tomo_spec=base_spec)
fixed_curves = ordered_curves(base_result.bins)

panel_curves = {
    "frac": [],
    "offset": [],
    "scale": [],
    "scatter": [],
}

for val in outlier_frac_vals:
    spec = make_spec(
        n_bins=n_bins,
        bin_range=bin_range,
        scatter_scale=[baseline_scatter, baseline_scatter, baseline_scatter],
        mean_scale=[baseline_mean_scale, baseline_mean_scale, baseline_mean_scale],
        mean_offset=[baseline_mean_offset, baseline_mean_offset, baseline_mean_offset],
        outlier_frac=[0.0, 0.0, float(val)],
        outlier_scatter_scale=[
            baseline_outlier_scatter,
            baseline_outlier_scatter,
            baseline_outlier_scatter,
        ],
        outlier_mean_scale=[
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
        ],
        outlier_mean_offset=[
            baseline_outlier_mean_offset,
            baseline_outlier_mean_offset,
            baseline_outlier_mean_offset,
        ],
    )
    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)
    panel_curves["frac"].append(
        [fixed_curves[0].copy(), fixed_curves[1].copy(), curves[2].copy()]
    )

for val in outlier_mean_offset_vals:
    spec = make_spec(
        n_bins=n_bins,
        bin_range=bin_range,
        scatter_scale=[baseline_scatter, baseline_scatter, baseline_scatter],
        mean_scale=[baseline_mean_scale, baseline_mean_scale, baseline_mean_scale],
        mean_offset=[baseline_mean_offset, baseline_mean_offset, baseline_mean_offset],
        outlier_frac=[0.0, 0.0, 0.55],
        outlier_scatter_scale=[
            baseline_outlier_scatter,
            baseline_outlier_scatter,
            baseline_outlier_scatter,
        ],
        outlier_mean_scale=[
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
        ],
        outlier_mean_offset=[0.0, 0.0, float(val)],
    )
    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)
    panel_curves["offset"].append(
        [fixed_curves[0].copy(), fixed_curves[1].copy(), curves[2].copy()]
    )

for val in outlier_mean_scale_vals:
    spec = make_spec(
        n_bins=n_bins,
        bin_range=bin_range,
        scatter_scale=[baseline_scatter, baseline_scatter, baseline_scatter],
        mean_scale=[baseline_mean_scale, baseline_mean_scale, baseline_mean_scale],
        mean_offset=[baseline_mean_offset, baseline_mean_offset, baseline_mean_offset],
        outlier_frac=[0.0, 0.0, 0.20],
        outlier_scatter_scale=[
            baseline_outlier_scatter,
            baseline_outlier_scatter,
            baseline_outlier_scatter,
        ],
        outlier_mean_scale=[1.0, 1.0, float(val)],
        outlier_mean_offset=[0.0, 0.0, baseline_outlier_mean_offset],
    )
    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)
    panel_curves["scale"].append(
        [fixed_curves[0].copy(), fixed_curves[1].copy(), curves[2].copy()]
    )

for val in outlier_scatter_vals:
    spec = make_spec(
        n_bins=n_bins,
        bin_range=bin_range,
        scatter_scale=[baseline_scatter, baseline_scatter, baseline_scatter],
        mean_scale=[baseline_mean_scale, baseline_mean_scale, baseline_mean_scale],
        mean_offset=[baseline_mean_offset, baseline_mean_offset, baseline_mean_offset],
        outlier_frac=[0.0, 0.0, 0.20],
        outlier_scatter_scale=[
            baseline_outlier_scatter,
            baseline_outlier_scatter,
            float(val),
        ],
        outlier_mean_scale=[
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
            baseline_outlier_mean_scale,
        ],
        outlier_mean_offset=[0.0, 0.0, baseline_outlier_mean_offset],
    )
    result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)
    curves = ordered_curves(result.bins)
    panel_curves["scatter"].append(
        [fixed_curves[0].copy(), fixed_curves[1].copy(), curves[2].copy()]
    )

ymax_bins = 1.08 * max(
    np.max(curve)
    for family in panel_curves.values()
    for frame in family
    for curve in frame
)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
axes = np.asarray(axes)

lw = 2.0

panel_defs = [
    ("frac", axes[0, 0], r"Outlier fraction", outlier_frac_vals, r"$f_{\rm out}$"),
    (
        "offset",
        axes[0, 1],
        r"Outlier mean offset",
        outlier_mean_offset_vals,
        r"$\beta_{\rm out}$",
    ),
    (
        "scale",
        axes[1, 0],
        r"Outlier mean scaling",
        outlier_mean_scale_vals,
        r"$\alpha_{\rm out}$",
    ),
    (
        "scatter",
        axes[1, 1],
        r"Outlier scatter",
        outlier_scatter_vals,
        r"$s_{\rm out}$",
    ),
]

panel_artists = {}
zero_vertices = fill_vertices(z, np.zeros_like(z))

for key, ax, title, values, symbol in panel_defs:
    ax.set_title(title)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, ymax_bins)

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
    )

    ax.grid(False)
    ax.plot(z, np.zeros_like(z), color="k", linewidth=lw, zorder=100)

    fills = []
    lines = []

    for color in colors:
        poly = Polygon(
            zero_vertices.copy(),
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.6,
            zorder=10,
            animated=True,
        )
        ax.add_patch(poly)
        fills.append(poly)

        (line,) = ax.plot(
            z,
            np.zeros_like(z),
            color="k",
            linewidth=lw,
            zorder=20,
            animated=True,
        )
        lines.append(line)

    text_main = ax.text(
        0.64,
        0.93,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        animated=True,
    )

    # slightly larger and a bit more left/lower so it reads more clearly
    proxy_x_center = 1.51
    proxy_halfwidth = 0.155
    proxy_y_base = 0.60 * ymax_bins
    proxy_y_amp = 0.14 * ymax_bins

    proxy_u = np.linspace(-3.2, 3.2, 280)

    # keep decomposition, but fade the unchanged components more
    (core_line,) = ax.plot(
        [],
        [],
        color="0.72",
        linewidth=1.2,
        linestyle="--",
        zorder=46,
        animated=True,
    )
    (outlier_line,) = ax.plot(
        [],
        [],
        color="0.35",
        linewidth=1.3,
        linestyle="--",
        zorder=47,
        animated=True,
    )
    (mix_line,) = ax.plot(
        [],
        [],
        color="k",
        linewidth=2.3,
        zorder=48,
        animated=True,
    )

    panel_artists[key] = {
        "ax": ax,
        "fills": fills,
        "lines": lines,
        "text": text_main,
        "values": values,
        "symbol": symbol,
        "proxy_u": proxy_u,
        "proxy_x_center": proxy_x_center,
        "proxy_halfwidth": proxy_halfwidth,
        "proxy_y_base": proxy_y_base,
        "proxy_y_amp": proxy_y_amp,
        "core_line": core_line,
        "outlier_line": outlier_line,
        "mix_line": mix_line,
    }

axes[0, 0].set_ylabel(r"$n_i(z)$")
axes[1, 0].set_ylabel(r"$n_i(z)$")
axes[1, 0].set_xlabel("Redshift $z$")
axes[1, 1].set_xlabel("Redshift $z$")


def draw_panel(key, frame_idx):
    artists = panel_artists[key]
    curves = panel_curves[key][frame_idx]

    for b in range(n_bins):
        y = curves[b]
        artists["fills"][b].set_xy(fill_vertices(z, y))
        artists["lines"][b].set_data(z, y)

    val = artists["values"][frame_idx]
    artists["text"].set_text(rf"{artists['symbol']} = {val:.3f}")

    u = artists["proxy_u"]

    core_center = 0.0
    core_sigma = 0.78

    if key == "frac":
        f_out = float(val)
        out_center = -1.75
        out_sigma = 0.60

    elif key == "offset":
        f_out = 0.55
        t = (val - outlier_mean_offset_vals.min()) / (
                outlier_mean_offset_vals.max() - outlier_mean_offset_vals.min()
        )
        out_center = 0.10 - 2.55 * t
        out_sigma = 0.62

    elif key == "scale":
        f_out = 0.20
        t = (val - outlier_mean_scale_vals.min()) / (
                outlier_mean_scale_vals.max() - outlier_mean_scale_vals.min()
        )
        out_center = -0.28 - 2.85 * (t ** 1.15)
        out_sigma = 0.55

    elif key == "scatter":
        f_out = 0.20
        out_center = -1.70
        t = (val - outlier_scatter_vals.min()) / (
                outlier_scatter_vals.max() - outlier_scatter_vals.min()
        )
        out_sigma = 0.22 + 1.55 * t

    core_profile = gaussian_profile(u, core_center, core_sigma)
    out_profile = gaussian_profile(u, out_center, out_sigma)
    mix_profile = (1.0 - f_out) * core_profile + f_out * out_profile

    peak = max(
        np.max(core_profile),
        np.max(out_profile),
        np.max(mix_profile),
    )
    core_profile /= peak
    out_profile /= peak
    mix_profile /= peak

    x_proxy = artists["proxy_x_center"] + artists["proxy_halfwidth"] * (u / 3.2)

    y_core = artists["proxy_y_base"] + artists["proxy_y_amp"] * core_profile
    y_out = artists["proxy_y_base"] + artists["proxy_y_amp"] * out_profile
    y_mix = artists["proxy_y_base"] + artists["proxy_y_amp"] * mix_profile

    artists["core_line"].set_data(x_proxy, y_core)
    artists["outlier_line"].set_data(x_proxy, y_out)
    artists["mix_line"].set_data(x_proxy, y_mix)

    return [
        *artists["fills"],
        *artists["lines"],
        artists["text"],
        artists["core_line"],
        artists["outlier_line"],
        artists["mix_line"],
    ]


def draw_frame(frame_idx):
    out = []
    for key, _, _, _, _ in panel_defs:
        out.extend(draw_panel(key, frame_idx))
    return out


def init():
    return draw_frame(frame_ids[0])


def update(i):
    return draw_frame(frame_ids[i])


plt.tight_layout()

anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_ids),
    init_func=init,
    interval=1000 / FPS,
    blit=True,
    repeat=True,
)

anim.save(OUTFILE, writer=PillowWriter(fps=FPS))
plt.close(fig)

print(f"Saved animation to: {OUTFILE}")