.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin summaries
====================

Once tomographic bins have been built, a good practice is to inspect
their statistical properties before using them in a forecast or analysis.

Binny provides several summary statistics through
:class:`binny.NZTomography` that describe both the **shape of the bin
curves** and the **distribution of galaxies across bins**. These
summaries help diagnose whether a tomographic binning scheme produces
well-separated, well-populated bins before the bins are used in a
forecast or cosmological analysis.

This page illustrates these summaries for a simple four-bin
photometric example and compares two common binning schemes:

- **equipopulated binning**, where each bin contains a similar fraction
  of the galaxy sample,
- **equidistant binning**, where the redshift interval is divided into
  bins of similar width.

The examples below focus on two families of summaries:

- **shape statistics**, such as bin centers, widths, quantiles, and peaks,
- **population statistics**, such as the fraction of galaxies per bin.

All plotting examples below are executable via ``.. plot::``.


Building a representative photo-z example
-----------------------------------------

We begin with a smooth parent redshift distribution and construct two
photometric tomographic realizations of the same sample: one using
equipopulated bins and one using equidistant bins.

Both cases use the same photo-z uncertainty model, so the comparison
isolates the effect of the binning strategy itself.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
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
               linewidth=2.0,
               zorder=20 + i,
           )

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
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

   equipopulated_result = NZTomography().build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   equidistant_result = NZTomography().build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, equipopulated_result.bins, "Equipopulated bins")
   axes[0].set_ylabel(r"Normalized $n_i(z)$")

   plot_bins(axes[1], z, equidistant_result.bins, "Equidistant bins")

   plt.tight_layout()


Accessing shape and population summaries
----------------------------------------

Shape and population summaries are typically inspected together when
evaluating a tomographic binning strategy.

**Shape statistics** describe the internal structure of each bin curve,
including representative redshift centers, widths, quantiles, and peak
locations. These quantities characterize how each bin samples the
underlying galaxy population.

**Population statistics** instead describe how galaxies are distributed
across bins, for example the fraction of the total sample assigned to
each bin or the corresponding galaxy number densities.

Together these summaries provide a compact diagnostic of whether the
chosen binning scheme produces balanced and well-behaved tomographic
bins.

.. code-block:: python

   import numpy as np

   from binny import NZTomography

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
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

   tomo_equipopulated = NZTomography()
   tomo_equipopulated.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   tomo_equidistant = NZTomography()
   tomo_equidistant.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   shape_equipopulated = tomo_equipopulated.shape_stats(
       center_method="median",
       decimal_places=3,
   )
   population_equipopulated = tomo_equipopulated.population_stats(
       density_total=30.0,
       decimal_places=3,
   )

   shape_equidistant = tomo_equidistant.shape_stats(
       center_method="median",
       decimal_places=3,
   )
   population_equidistant = tomo_equidistant.population_stats(
       density_total=30.0,
       decimal_places=3,
   )

   print("Equipopulated shape statistics")
   print(shape_equipopulated)
   print()
   print("Equipopulated population statistics")
   print(population_equipopulated)
   print()
   print("Equidistant shape statistics")
   print(shape_equidistant)
   print()
   print("Equidistant population statistics")
   print(population_equidistant)


Visualizing representative bin centers
--------------------------------------

A compact way to compare the two binning schemes is to reduce each bin
to a small set of representative redshift summaries, such as the
**mean**, **median**, and **mode**.

Although these summaries do not capture the full bin shape, they provide
a simple way to compare how different binning schemes place their bins
across the redshift range of the parent sample and how sensitive the
notion of a “bin center” is to the chosen definition.

Representative bin centers also play an important role in many
cosmological analyses. In weak lensing, galaxy clustering, and
galaxy-galaxy lensing forecasts, several astrophysical systematics are
often parameterized per tomographic bin, with the parameter values
defined as functions of the bin center. Examples include intrinsic
alignment amplitudes, galaxy bias, and magnification bias, which are
commonly evaluated at a characteristic redshift associated with each
bin.

For this reason, inspecting different definitions of bin centers can be
useful when assessing how tomographic binning choices propagate into
downstream modeling assumptions.


.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import NZTomography

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
       "scatter_scale": [0.06, 0.08, 0.10, 0.12],
       "mean_offset": [0.03, 0.05, 0.07, 0.10],
       "outlier_frac": [0.08, 0.10, 0.12, 0.15],
       "outlier_scatter_scale": [0.30, 0.35, 0.40, 0.45],
       "outlier_mean_offset": [0.15, 0.20, 0.25, 0.30],
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

   tomo_equipopulated = NZTomography()
   tomo_equipopulated.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   tomo_equidistant = NZTomography()
   tomo_equidistant.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   center_methods = ["mean", "median", "mode"]

   centers_equipopulated = {
       method: tomo_equipopulated.shape_stats(
           center_method=method,
           decimal_places=3,
       )["centers"]
       for method in center_methods
   }

   centers_equidistant = {
       method: tomo_equidistant.shape_stats(
           center_method=method,
           decimal_places=3,
       )["centers"]
       for method in center_methods
   }

   keys = sorted(centers_equipopulated["mean"].keys())
   x = np.arange(len(keys))

   colors = cmr.take_cmap_colors(
       "viridis",
       3,
       cmap_range=(0.0, 1.0),
       return_fmt="hex",
   )

   fill_colors = {
       m: to_rgba(c, 0.6)
       for m, c in zip(center_methods, colors)
   }

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

   fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)

   scheme_data = [
       (axes[0], "Equipopulated bins", centers_equipopulated),
       (axes[1], "Equidistant bins", centers_equidistant),
   ]

   for ax, title, centers in scheme_data:

       for method in center_methods:
           ax.scatter(
               x + offset_map[method],
               [centers[method][key] for key in keys],
               marker=marker_map[method],
               s=200,
               facecolors=fill_colors[method],
               edgecolors="k",
               linewidths=1.6,
               label=label_map[method],
               zorder=3,
           )

       ax.set_title(title)
       ax.set_xticks(x)
       ax.set_xticklabels([f"{key+1}" for key in keys])
       ax.set_xlabel("Tomographic bin")
       ax.legend(frameon=True)

   axes[0].set_ylabel("Representative redshift")

   plt.tight_layout()


Comparing per-bin galaxy fractions
----------------------------------

Population fractions are especially useful when comparing different
binning schemes. Equipopulated binning is designed to place a similar
fraction of galaxies into each bin, whereas equidistant binning is
instead controlled by redshift width.

In practice, the equipopulated fractions may not be exactly identical,
because photometric redshift uncertainties scatter galaxies across bin
boundaries and slightly modify the final observed-bin populations.

The bar chart below compares the resulting per-bin fractions.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import NZTomography

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
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

   tomo_equipopulated = NZTomography()
   tomo_equipopulated.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   tomo_equidistant = NZTomography()
   tomo_equidistant.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   fractions_equipopulated = tomo_equipopulated.population_stats(decimal_places=4)["fractions"]
   fractions_equidistant = tomo_equidistant.population_stats(decimal_places=4)["fractions"]

   keys = sorted(fractions_equipopulated.keys())
   x = np.arange(len(keys))
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

   plt.figure(figsize=(7.8, 4.9))
   plt.bar(
       x - width / 2,
       [fractions_equipopulated[key] for key in keys],
       width=width,
       color=fill_eqpop,
       edgecolor="k",
       linewidth=2.5,
       label="Equipopulated",
   )
   plt.bar(
       x + width / 2,
       [fractions_equidistant[key] for key in keys],
       width=width,
       color=fill_eqdist,
       edgecolor="k",
       linewidth=2.5,
       label="Equidistant",
   )

   plt.xticks(x, [f"{key+1}" for key in keys])
   plt.yticks()
   plt.xlabel("Tomographic bin")
   plt.ylabel("Galaxy fraction")
   plt.title("Per-bin population fractions")
   plt.legend(frameon=False)
   plt.tight_layout()


Comparing widths and peak locations
-----------------------------------

In addition to representative centers, it is often useful to compare the
effective width and characteristic peak location of each tomographic bin.

The **central 68% width** provides a robust summary of the redshift spread
within a bin. Unlike a simple standard deviation, it is directly tied to
the inner percentile range of the distribution and is therefore easier to
interpret for skewed or slightly non-Gaussian bin shapes. Narrower widths
usually indicate better redshift localization, while broader widths can
signal stronger photometric smearing or more leakage across nominal bin
boundaries.

The **peak-location summaries** describe where the bin is most strongly
concentrated in redshift. Here we compare the **mean**, **median**, and
**mode** for each bin. Looking at these together is useful because they
respond differently to skewness, extended tails, and secondary structure.
If these three summaries lie close together, the bin is usually fairly
compact and symmetric. If they separate, that often indicates skewness,
leakage, or outlier-driven distortions in the bin shape.

These quantities are useful in cosmological investigations because many
downstream ingredients are attached to a characteristic bin redshift.
Examples include intrinsic-alignment amplitudes, galaxy-bias parameters,
magnification-bias terms, and other nuisance or astrophysical quantities
that are commonly defined per tomographic bin. At the same time, the bin
width controls how broadly the galaxy sample projects the underlying
matter field along the line of sight. Inspecting both width and peak
summaries therefore helps assess how binning choices may propagate into
signal modeling, parameter interpretation, and the level of overlap
between neighboring bins.

The figure below shows the central 68\% widths as a histogram-style bar
comparison and the corresponding mean, median, and mode values for each
bin.


.. plot::
   :include-source: False
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import NZTomography

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
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
   tomo_eqpop.build_bins(z=z, nz=nz, tomo_spec=equipopulated_spec)

   tomo_eqdist = NZTomography()
   tomo_eqdist.build_bins(z=z, nz=nz, tomo_spec=equidistant_spec)

   shape_eqpop = tomo_eqpop.shape_stats(center_method="median")
   shape_eqdist = tomo_eqdist.shape_stats(center_method="median")

   keys = sorted(shape_eqpop["per_bin"].keys())
   x = np.arange(len(keys))
   width = 0.36

   widths_eqpop = [
       shape_eqpop["per_bin"][k]["moments"]["width_68"] for k in keys
   ]
   widths_eqdist = [
       shape_eqdist["per_bin"][k]["moments"]["width_68"] for k in keys
   ]

   center_methods = ["mean", "median", "mode"]

   centers_eqpop = {
       m: tomo_eqpop.shape_stats(center_method=m)["centers"] for m in center_methods
   }

   centers_eqdist = {
       m: tomo_eqdist.shape_stats(center_method=m)["centers"] for m in center_methods
   }

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   c_eqpop, c_eqdist = colors
   fill_eqpop = to_rgba(c_eqpop, 0.6)
   fill_eqdist = to_rgba(c_eqdist, 0.6)

   marker_map = {
       "mean": "o",
       "median": "s",
       "mode": "v",
   }

   offset_map = {
       "mean": -0.18,
       "median": 0.0,
       "mode": 0.18,
   }

   fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))

   axes[0].bar(
       x - width/2,
       widths_eqpop,
       width=width,
       color=fill_eqpop,
       edgecolor="k",
       linewidth=2,
       label="Equipopulated",
   )

   axes[0].bar(
       x + width/2,
       widths_eqdist,
       width=width,
       color=fill_eqdist,
       edgecolor="k",
       linewidth=2,
       label="Equidistant",
   )

   axes[0].set_title("Central 68% widths")
   axes[0].set_xlabel("Tomographic bin")
   axes[0].set_ylabel("Width in redshift")
   axes[0].set_xticks(x)
   axes[0].set_xticklabels([f"{k+1}" for k in keys])
   axes[0].legend(frameon=False)

   for m in center_methods:

       axes[1].scatter(
           x + offset_map[m] - 0.02,
           [centers_eqpop[m][k] for k in keys],
           marker=marker_map[m],
           s=160,
           color=fill_eqpop,
           edgecolor="k",
           linewidth=1.5,
           label=f"{m.capitalize()} (Equipopulated)",
       )

       axes[1].scatter(
           x + offset_map[m] + 0.02,
           [centers_eqdist[m][k] for k in keys],
           marker=marker_map[m],
           s=160,
           color=fill_eqdist,
           edgecolor="k",
           linewidth=1.5,
           label=f"{m.capitalize()} (Equidistant)",
       )

   axes[1].set_title("Peak-location summaries")
   axes[1].set_xlabel("Tomographic bin")
   axes[1].set_ylabel("Representative redshift")
   axes[1].set_xticks(x)
   axes[1].set_xticklabels([f"{k+1}" for k in keys])
   axes[1].legend(frameon=True)

   plt.tight_layout()


Tail asymmetry per bin
----------------------

Tail asymmetry compares the upper and lower spread around the median,
using the 16th, 50th, and 84th percentiles.

Values above unity indicate that the distribution extends further toward
higher redshift than toward lower redshift, while values below unity
indicate the opposite. Strong asymmetries can signal leakage from
neighboring bins or skewness introduced by the photo-z model.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
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

   tomo_equipopulated = NZTomography()
   tomo_equipopulated.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   tomo_equidistant = NZTomography()
   tomo_equidistant.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   shape_equipopulated = tomo_equipopulated.shape_stats(
       center_method="median",
       decimal_places=4,
   )
   shape_equidistant = tomo_equidistant.shape_stats(
       center_method="median",
       decimal_places=4,
   )

   keys = sorted(shape_equipopulated["per_bin"].keys())
   x = np.arange(len(keys))
   width = 0.36

   values_equipopulated = [
       shape_equipopulated["per_bin"][key]["tail_asymmetry"]
       for key in keys
   ]
   values_equidistant = [
       shape_equidistant["per_bin"][key]["tail_asymmetry"]
       for key in keys
   ]

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   plt.figure(figsize=(7.8, 4.8))
   plt.bar(
       x - width / 2,
       values_equipopulated,
       width=width,
       color=colors[0],
       edgecolor="k",
       linewidth=1.2,
       label="Equipopulated",
   )
   plt.bar(
       x + width / 2,
       values_equidistant,
       width=width,
       color=colors[1],
       edgecolor="k",
       linewidth=1.2,
       label="Equidistant",
   )

   plt.axhline(1.0, linewidth=1.4)
   plt.xticks(x, [f"Bin {key}" for key in keys])
   plt.ylabel("Tail asymmetry")
   plt.title("Tail asymmetry by bin")
   plt.legend(frameon=False)
   plt.tight_layout()


Secondary peak strength
-----------------------

Secondary peak strength measures how important the second-highest peak
is relative to the primary peak in each bin curve.

Larger values indicate stronger multimodality or more pronounced
secondary structure, which can arise from photo-z outliers or leakage
from neighboring bins.

A nonzero value does not necessarily indicate a pathological binning
scheme. In many cases it simply shows that a subset of galaxies is being
scattered far enough from the main concentration to form a distinct
secondary bump. This can happen more strongly in some bins than in
others, depending on the local bin width, the shape of the parent
distribution, and the adopted uncertainty model.

In practice, this diagnostic is most useful as a flag for bins that may
deserve closer inspection. Small values usually indicate weak secondary
structure, whereas larger values suggest that the bin contains a more
clearly separated subpopulation and may be more strongly affected by
outliers or cross-bin leakage.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   z = np.linspace(0.0, 2.0, 1500)

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
       "outlier_frac": [0.20, 0.22, 0.24, 0.26],
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

   tomo_equipopulated = NZTomography()
   tomo_equipopulated.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
       include_tomo_metadata=True,
   )

   tomo_equidistant = NZTomography()
   tomo_equidistant.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   shape_equipopulated = tomo_equipopulated.shape_stats(
       center_method="median",
       decimal_places=4,
   )
   shape_equidistant = tomo_equidistant.shape_stats(
       center_method="median",
       decimal_places=4,
   )

   keys = sorted(shape_equipopulated["per_bin"].keys())
   x = np.arange(len(keys))
   width = 0.36

   values_equipopulated = [
       shape_equipopulated["per_bin"][key]["peaks"]["second_peak_ratio"] or 0.0
       for key in keys
   ]
   values_equidistant = [
       shape_equidistant["per_bin"][key]["peaks"]["second_peak_ratio"] or 0.0
       for key in keys
   ]

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   plt.figure(figsize=(7.8, 4.8))
   plt.bar(
       x - width / 2,
       values_equipopulated,
       width=width,
       color=colors[0],
       edgecolor="k",
       linewidth=1.2,
       label="Equipopulated",
   )
   plt.bar(
       x + width / 2,
       values_equidistant,
       width=width,
       color=colors[1],
       edgecolor="k",
       linewidth=1.2,
       label="Equidistant",
   )

   plt.xticks(x, [f"Bin {key}" for key in keys])
   plt.ylabel("Second peak / first peak")
   plt.title("Secondary peak strength")
   plt.legend(frameon=False)
   plt.tight_layout()


In-range fraction per bin
-------------------------

The in-range fraction measures how much of each bin curve lies inside
its own nominal redshift interval.

This provides a direct summary of how well each bin remains confined to
the redshift interval it was intended to represent. Lower values indicate
stronger leakage outside the nominal bin boundaries.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

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
       "scatter_scale": 0.05,
       "mean_offset": 0.01,
       "outlier_frac": 0.03,
       "outlier_scatter_scale": 0.20,
       "outlier_mean_offset": 0.05,
   }

   equidistant_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equidistant", "n_bins": 4, "range": (0.2, 1.2)},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   bin_edges = [0.2, 0.45, 0.70, 0.95, 1.20]

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
       include_tomo_metadata=True,
   )

   shape = tomo.shape_stats(
       center_method="median",
       decimal_places=4,
       bin_edges=bin_edges,
   )

   fractions = shape["in_range_fraction"]
   keys = sorted(fractions.keys())
   x = np.arange(len(keys))

   colors = cmr.take_cmap_colors(
       "viridis",
       len(keys),
       cmap_range=(0.1, 0.9),
       return_fmt="hex",
   )

   plt.figure(figsize=(7.6, 4.7))
   plt.bar(
       x,
       [100.0 * fractions[key] for key in keys],
       color=colors,
       edgecolor="k",
       linewidth=1.2,
   )
   plt.xticks(x, [f"Bin {key}" for key in keys])
   plt.ylabel("In-range fraction [%]")
   plt.title("Fraction of each bin inside its nominal range")
   plt.tight_layout()


Nominal bin widths
------------------

Nominal bin widths provide a direct summary of the binning strategy.

Equipopulated binning typically produces bins with different redshift
widths because the edges are chosen to equalize the galaxy counts,
whereas equidistant binning keeps the redshift intervals similar but
allows the number of galaxies per bin to vary.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   equipopulated_edges = np.array([0.20, 0.33, 0.49, 0.72, 1.20])
   equidistant_edges = np.array([0.20, 0.45, 0.70, 0.95, 1.20])

   widths_equipopulated = np.diff(equipopulated_edges)
   widths_equidistant = np.diff(equidistant_edges)

   x = np.arange(len(widths_equipopulated))
   width = 0.36

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   plt.figure(figsize=(7.8, 4.8))
   plt.bar(
       x - width / 2,
       widths_equipopulated,
       width=width,
       color=colors[0],
       edgecolor="k",
       linewidth=1.2,
       label="Equipopulated",
   )
   plt.bar(
       x + width / 2,
       widths_equidistant,
       width=width,
       color=colors[1],
       edgecolor="k",
       linewidth=1.2,
       label="Equidistant",
   )

   plt.xticks(x, [f"Bin {i}" for i in x])
   plt.ylabel("Nominal width in redshift")
   plt.title("Nominal bin widths")
   plt.legend(frameon=False)
   plt.tight_layout()


Notes
-----

- Shape statistics summarize the structure of the bin curves themselves
  and are safe to compute even when each bin is normalized.
- Population statistics depend on tomography metadata and therefore
  require rebuilding with ``include_tomo_metadata=True``.
- Equipopulated and equidistant binning can produce noticeably different
  bin centers, widths, and population fractions, even when they are
  built from the same parent distribution and uncertainty model.
- The summaries returned by :class:`binny.NZTomography` are ordinary
  Python dictionaries, so the quantities shown here can also be saved
  or reused directly in later analysis steps.