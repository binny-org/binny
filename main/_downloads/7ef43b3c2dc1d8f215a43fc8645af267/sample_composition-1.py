import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography


lrg_tomo = NZTomography()
elg_tomo = NZTomography()

lrg = lrg_tomo.build_survey_bins(
    "desi",
    role="lens",
    sample="lrg",
    overrides={"bins": {"edges": [0.4, 0.6, 0.8, 1.0]}},
)

elg = elg_tomo.build_survey_bins(
    "desi",
    role="lens",
    sample="elg",
    overrides={"bins": {"edges": [0.6, 0.9, 1.2, 1.5]}},
)

z_combined, nz_combined = NZTomography.combine_parent_nz(
    [
        {"z": lrg.z, "nz": lrg.nz},
        {"z": elg.z, "nz": elg.nz},
    ],
    interpolate=True,
)

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.15, 0.85),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.fill_between(
    lrg.z,
    0.0,
    lrg.nz,
    color=colors[0],
    alpha=0.45,
    linewidth=0.0,
    label="LRG parent",
)
ax.plot(lrg.z, lrg.nz, color="k", linewidth=1.8)

ax.fill_between(
    elg.z,
    0.0,
    elg.nz,
    color=colors[1],
    alpha=0.45,
    linewidth=0.0,
    label="ELG parent",
)
ax.plot(elg.z, elg.nz, color="k", linewidth=1.8)

ax.plot(
    z_combined,
    nz_combined,
    color=colors[2],
    linewidth=3.0,
    label="Combined parent",
)

ax.plot(z_combined, np.zeros_like(z_combined), color="k", linewidth=2.0)

ax.set_xlim(0.35, 1.55)
ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")
ax.set_title("DESI parent sample composition")
ax.legend(frameon=False)

plt.tight_layout()