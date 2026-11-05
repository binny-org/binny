import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from binny import NZTomography

z = np.linspace(0.0, 2.0, 1500)

nz = (
    np.exp(-0.5 * ((z - 0.6) / 0.12) ** 2)
    + 0.55 * np.exp(-0.5 * ((z - 1.4) / 0.15) ** 2)
)

nz /= np.trapezoid(nz, x=z)

result = NZTomography().build_predefined_bins(
    z=z,
    nz=nz,
    bins={0: nz},
)

peaks = result.parent_stats(
    min_rel_height=0.1,
    decimal_places=None,
)["peaks"]

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

mode_color, _, peak_color = colors

fig, ax = plt.subplots(figsize=(8.0, 4.8))

ax.fill_between(
    z,
    0.0,
    nz,
    color=to_rgba(peak_color, 0.6),
    linewidth=0.0,
)

ax.plot(
    z,
    nz,
    color="k",
    linewidth=2.5,
    label=r"Parent $n(z)$",
)

ax.axvline(
    peaks["mode"],
    color=mode_color,
    linestyle="-",
    linewidth=2.0,
    label="Global mode",
)

ax.plot(
    z,
    np.zeros_like(z),
    color="k",
    linewidth=2.2,
)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title(
    f"Parent distribution with {int(peaks['num_peaks'])} significant peaks"
)
ax.legend(frameon=False)

plt.tight_layout()