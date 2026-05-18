import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

equipopulated_edges = np.array([0.20, 0.33, 0.49, 0.72, 1.20])
equidistant_edges = np.array([0.20, 0.45, 0.70, 0.95, 1.20])

widths_equipopulated = np.diff(equipopulated_edges)
widths_equidistant = np.diff(equidistant_edges)

x = np.arange(len(widths_equipopulated))
width = 0.36

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.0, 1.0),
    return_fmt="hex",
)

_, c_eqpop, c_eqdist = colors
fill_eqpop = to_rgba(c_eqpop, 0.6)
fill_eqdist = to_rgba(c_eqdist, 0.6)

plt.figure(figsize=(7.8, 4.8))
plt.bar(
    x - width / 2,
    widths_equipopulated,
    width=width,
    color=fill_eqpop,
    edgecolor="k",
    linewidth=2.5,
    label="Equipopulated",
)
plt.bar(
    x + width / 2,
    widths_equidistant,
    width=width,
    color=fill_eqdist,
    edgecolor="k",
    linewidth=2.5,
    label="Equidistant",
)

plt.xticks(x, [f"{i+1}" for i in x])
plt.xlabel("Tomographic bin")
plt.ylabel("Nominal width in redshift")
plt.title("Nominal bin widths by scheme")
plt.legend(frameon=False)
plt.tight_layout()