import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from binny import NZTomography

z = np.linspace(0.0, 2.0, 1000)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.2,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

result = NZTomography().build_predefined_bins(
    z=z,
    nz=nz,
    bins={0: nz},
)

stats = result.parent_stats(decimal_places=None)

q16 = stats["quantiles"][0.16]
q50 = stats["quantiles"][0.50]
q84 = stats["quantiles"][0.84]

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

_, interval_color, median_color = colors

fig, ax = plt.subplots(figsize=(8.0, 4.8))

ax.plot(
    z,
    nz,
    color="k",
    linewidth=2.5,
    label=r"Parent $n(z)$",
)

ax.fill_between(
    z,
    0.0,
    nz,
    where=(z >= q16) & (z <= q84),
    color=to_rgba(interval_color, 0.6),
    linewidth=0.0,
    label="Central 68% interval",
)

ax.axvline(
    q16,
    color=interval_color,
    linestyle="-",
    linewidth=2.0,
)

ax.axvline(
    q84,
    color=interval_color,
    linestyle="-",
    linewidth=2.0,
)

ax.axvline(
    q50,
    color=median_color,
    linestyle="-",
    linewidth=2.0,
    label="Median",
)

ax.plot(
    z,
    np.zeros_like(z),
    color="k",
    linewidth=2.2,
)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("Parent redshift quantiles")
ax.legend(frameon=False)

plt.tight_layout()