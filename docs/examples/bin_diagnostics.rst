.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin diagnostics
======================

Once tomographic bins have been built, it is often useful to inspect
their statistical properties before using them in a forecast or analysis.

Binny provides several diagnostics through :class:`binny.NZTomography`
that quantify three complementary aspects of a tomographic binning
scheme: the **shape of the bin curves**, the **distribution of galaxies
across bins**, and the **degree of coupling between bins** through
overlap, leakage, or correlated structure. Together, these diagnostics
help assess whether the resulting bins are well separated, well
populated, and suitable for downstream forecasting or cosmological
analysis.

This page illustrates these diagnostics for a simple four-bin
photometric example and compares two common binning schemes:

- **equipopulated binning**, where each bin contains a similar fraction
  of the galaxy sample,
- **equidistant binning**, where the redshift interval is divided into
  bins of similar width.

The examples below focus on three families of diagnostics, each probing
a different aspect of tomographic-bin quality:

- **shape statistics**, such as bin centers, widths, quantiles, and peaks,
- **population statistics**, such as the fraction of galaxies per bin,
- **cross-bin diagnostics**, such as overlap, leakage, and Pearson correlation.

The diagnostics on this page compare bins within a single tomographic
sample. In joint analyses, it is often also useful to compare bins
between different samples, for example lens and source bins in
galaxy-galaxy lensing. Such between-sample diagnostics can be used to
quantify redshift separation, overlap, and potential foreground
contamination across sample families.

All examples below are executable via ``.. plot::``.


Building a representative photo-z example
-----------------------------------------

We begin with a smooth parent redshift distribution and construct two
photometric tomographic realizations of the same sample: one using
equipopulated bins and one using equidistant bins.

Both cases use the same photo-z uncertainty model, so differences in the
resulting diagnostics can be attributed primarily to the binning scheme
rather than to changes in the underlying uncertainty assumptions.

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
       ax.set_title(title, fontsize=15)
       ax.set_xlabel("Redshift $z$", fontsize=15)

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
   axes[0].set_ylabel(r"Normalized $n_i(z)$", fontsize=15)

   plot_bins(axes[1], z, equidistant_result.bins, "Equidistant bins")

   plt.tight_layout()



Cross-bin diagnostics
---------------------

In addition to per-bin summaries, it is often useful to quantify how
strongly different tomographic bins are coupled to one another. Binny
provides several cross-bin diagnostics for this purpose, including
**overlap measures**, **leakage matrices**, and **Pearson correlations**
between bin curves.

These metrics do not measure exactly the same thing:

- **overlap** quantifies how much two bin distributions occupy the same
  region of redshift space,
- **leakage** quantifies how much of the content of one bin falls inside
  the nominal redshift interval of another bin,
- **Pearson correlation** quantifies the similarity of bin shapes as
  sampled on the common redshift grid.

The figure below compares these diagnostics for a four-bin equidistant
photo-z example.

.. plot::
   :include-source: True
   :width: 900

   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def nested_dict_to_matrix(nested_dict):
       keys = sorted(nested_dict.keys())
       matrix = np.array(
           [[nested_dict[row_key][col_key] for col_key in keys] for row_key in keys],
           dtype=float,
       )
       return keys, matrix

   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   bin_edges = [0.2, 0.45, 0.70, 0.95, 1.20]

   stats = tomo.cross_bin_stats(
       overlap={"method": "min", "unit": "percent", "normalize": True, "decimal_places": 3},
       leakage={"bin_edges": bin_edges, "unit": "percent", "decimal_places": 3},
       pearson={"normalize": True, "decimal_places": 3},
   )

   overlap_keys, overlap_matrix = nested_dict_to_matrix(stats["overlap"])
   leakage_keys, leakage_matrix = nested_dict_to_matrix(stats["leakage"])
   pearson_keys, pearson_matrix = nested_dict_to_matrix(stats["pearson"])

   fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

   matrices = [
       (overlap_keys, overlap_matrix, "Overlap matrix", "Bin index", "Bin index"),
       (leakage_keys, leakage_matrix, "Leakage matrix", "Nominal bin", "Source bin"),
       (pearson_keys, pearson_matrix, "Pearson matrix", "Bin index", "Bin index"),
   ]

   for ax, (keys, matrix, title, xlabel, ylabel) in zip(axes, matrices, strict=True):
       ax.imshow(matrix, origin="lower", aspect="auto")
       ax.set_title(title, fontsize=15)
       ax.set_xticks(np.arange(len(keys)))
       ax.set_yticks(np.arange(len(keys)))
       ax.set_xticklabels(keys, fontsize=11)
       ax.set_yticklabels(keys, fontsize=11)
       ax.set_xlabel(xlabel, fontsize=13)
       ax.set_ylabel(ylabel, fontsize=13)

       for i in range(matrix.shape[0]):
           for j in range(matrix.shape[1]):
               ax.text(
                   j,
                   i,
                   f"{matrix[i, j]:.1f}",
                   ha="center",
                   va="center",
                   fontsize=9,
                   color="k",
               )

   plt.tight_layout()

In all three matrices, values near the diagonal typically indicate the
strongest self-association, while off-diagonal structure reveals
coupling between different bins. Large off-diagonal overlap or leakage
usually signals reduced bin separation, whereas large off-diagonal
Pearson correlation indicates that two bins have similar shapes even if
their normalization or interpretation differs.


Multi-metric cross-bin comparison
---------------------------------

Different cross-bin metrics emphasize different aspects of similarity or
separation between bins, so it is often useful to compare more than one
metric.

The figure below shows several commonly used pairwise measures for the
same tomographic bin set:

- **min overlap** measures the shared area between two normalized bin curves,
- **cosine similarity** measures how closely aligned two bin vectors are
  on the sampled redshift grid,
- **Jensen--Shannon distance** measures how different two normalized
  distributions are in an information-theoretic sense,
- **Hellinger distance** measures geometric dissimilarity between two
  probability distributions,
- **total variation distance** measures the maximum overall discrepancy
  between two normalized distributions.

Similarity-like measures are largest when bins are more alike, whereas
distance-like measures are smallest when bins are more alike. Comparing
them side by side helps identify which bin pairs are consistently close
across definitions and which depend more strongly on the chosen metric.

.. plot::
   :include-source: True
   :width: 980

   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def nested_dict_to_matrix(nested_dict):
       keys = sorted(nested_dict.keys())
       matrix = np.array(
           [[nested_dict[row_key][col_key] for col_key in keys] for row_key in keys],
           dtype=float,
       )
       return keys, matrix

   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   overlap_min = tomo.cross_bin_stats(
       overlap={"method": "min", "unit": "percent", "normalize": True, "decimal_places": 3},
   )["overlap"]

   overlap_cosine = tomo.cross_bin_stats(
       overlap={"method": "cosine", "unit": "percent", "normalize": False, "decimal_places": 3},
   )["overlap"]

   overlap_js = tomo.cross_bin_stats(
       overlap={"method": "js", "unit": "fraction", "normalize": True, "decimal_places": 3},
   )["overlap"]

   overlap_hellinger = tomo.cross_bin_stats(
       overlap={"method": "hellinger", "unit": "fraction", "normalize": True, "decimal_places": 3},
   )["overlap"]

   overlap_tv = tomo.cross_bin_stats(
       overlap={"method": "tv", "unit": "fraction", "normalize": True, "decimal_places": 3},
   )["overlap"]

   metric_specs = [
       ("Min overlap [%]", overlap_min),
       ("Cosine similarity [%]", overlap_cosine),
       ("JS distance", overlap_js),
       ("Hellinger distance", overlap_hellinger),
       ("TV distance", overlap_tv),
   ]

   fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
   axes = axes.ravel()

   for ax, (title, metric_dict) in zip(axes, metric_specs):
       keys, matrix = nested_dict_to_matrix(metric_dict)
       ax.imshow(matrix, origin="lower", aspect="auto")
       ax.set_title(title, fontsize=14)
       ax.set_xticks(np.arange(len(keys)))
       ax.set_yticks(np.arange(len(keys)))
       ax.set_xticklabels(keys, fontsize=11)
       ax.set_yticklabels(keys, fontsize=11)
       ax.set_xlabel("Bin index", fontsize=12)
       ax.set_ylabel("Bin index", fontsize=12)

       for i in range(matrix.shape[0]):
           for j in range(matrix.shape[1]):
               ax.text(
                   j,
                   i,
                   f"{matrix[i, j]:.2f}",
                   ha="center",
                   va="center",
                   fontsize=8,
                   color="k",
               )

   axes[-1].axis("off")

   plt.tight_layout()


Ranking the most overlapping bin pairs
--------------------------------------

A pair ranking provides a compact summary of which tomographic bin pairs
are most strongly coupled according to a chosen pairwise metric.

Here the ranking is based on **min overlap**, so pairs with larger values
share more support in redshift and are less cleanly separated. In many
practical cases, the most strongly overlapping pairs are neighboring
bins, since photo-z scatter tends to move galaxies preferentially into
adjacent redshift intervals.

The bar chart below shows the largest pairwise min-overlap values for a
four-bin photo-z example.

.. plot::
   :include-source: True
   :width: 760

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

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   pair_list = tomo.cross_bin_stats(
       pairs={
           "method": "min",
           "unit": "percent",
           "threshold": 0.0,
           "direction": "high",
           "normalize": True,
           "decimal_places": 3,
       }
   )["correlations"]

   labels = [f"({i}, {j})" for i, j, _ in pair_list]
   values = [value for _, _, value in pair_list]
   y = np.arange(len(labels))

   plt.figure(figsize=(7.8, 4.8))
   plt.barh(y, values, edgecolor="k", linewidth=1.2)
   plt.yticks(y, labels, fontsize=12)
   plt.xlabel("Min overlap [%]", fontsize=13)
   plt.title("Pairwise overlap ranking", fontsize=14)
   plt.gca().invert_yaxis()
   plt.tight_layout()


Completeness and contamination per bin
--------------------------------------

The leakage matrix can be summarized into per-bin **completeness** and
**contamination** measures.

For a given source bin, the diagonal leakage entry gives the fraction of
its weight that remains inside its own nominal redshift interval. This
is the **completeness** of that bin. Its complement measures the fraction
that falls outside the intended interval and therefore quantifies
**contamination** by leakage into other nominal bins.

Large completeness indicates that a bin remains well confined to its
target redshift range, while large contamination indicates stronger
mixing with neighboring bins.

.. plot::
   :include-source: True
   :width: 760

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

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   bin_edges = [0.2, 0.45, 0.70, 0.95, 1.20]

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   leakage = tomo.cross_bin_stats(
       leakage={"bin_edges": bin_edges, "unit": "percent", "decimal_places": 3},
   )["leakage"]

   keys = sorted(leakage.keys())
   completeness = [leakage[key][key] for key in keys]
   contamination = [100.0 - leakage[key][key] for key in keys]
   x = np.arange(len(keys))
   width = 0.38

   plt.figure(figsize=(7.8, 4.8))
   plt.bar(
       x - width / 2,
       completeness,
       width=width,
       edgecolor="k",
       linewidth=1.2,
       label="Completeness",
   )
   plt.bar(
       x + width / 2,
       contamination,
       width=width,
       edgecolor="k",
       linewidth=1.2,
       label="Contamination",
   )
   plt.xticks(x, [f"Bin {key}" for key in keys], fontsize=12)
   plt.ylabel("Percent", fontsize=13)
   plt.title("Leakage-based completeness and contamination", fontsize=14)
   plt.legend(frameon=False, fontsize=12)
   plt.tight_layout()


Leakage composition by nominal bin
----------------------------------

A stacked leakage view shows how the weight of each source bin is
distributed across the nominal redshift intervals.

Each bar corresponds to one source bin. The stacked segments show what
fraction of that bin lies inside each target nominal interval. The
diagonal component therefore represents self-consistency, while the
off-diagonal components reveal where leaked weight ends up.

This visualization is especially useful for identifying whether leakage
is mostly local, for example into adjacent bins, or whether there are
longer-range tails and outlier-driven transfers across more distant bins.

.. plot::
   :include-source: True
   :width: 820

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

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   bin_edges = [0.2, 0.45, 0.70, 0.95, 1.20]

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   leakage = tomo.cross_bin_stats(
       leakage={"bin_edges": bin_edges, "unit": "percent", "decimal_places": 3},
   )["leakage"]

   keys = sorted(leakage.keys())
   x = np.arange(len(keys))

   colors = cmr.take_cmap_colors(
       "viridis",
       len(keys),
       cmap_range=(0.1, 0.9),
       return_fmt="hex",
   )

   bottoms = np.zeros(len(keys))

   plt.figure(figsize=(8.2, 5.0))

   for color, target_key in zip(colors, keys, strict=True):
       values = [leakage[source_key][target_key] for source_key in keys]
       plt.bar(
           x,
           values,
           bottom=bottoms,
           color=color,
           edgecolor="k",
           linewidth=1.0,
           label=f"Nominal bin {target_key}",
       )
       bottoms += np.array(values)

   plt.xticks(x, [f"Source bin {key}" for key in keys], fontsize=12)
   plt.ylabel("Percent of source-bin mass", fontsize=13)
   plt.title("Leakage composition", fontsize=14)
   plt.legend(frameon=False, fontsize=10)
   plt.tight_layout()


Pearson correlation versus overlap
----------------------------------

Different pairwise metrics can show similar overall trends while still
capturing distinct features of the bin curves.

Here, **min overlap** measures how much two bins directly share support
in redshift, whereas **Pearson correlation** measures how similarly the
two sampled curves vary across the redshift grid. Two bins can therefore
have modest direct overlap but still exhibit a relatively strong shape
correlation, or vice versa.

The scatter plot below compares Pearson correlation against min-overlap
for all unique off-diagonal bin pairs.

.. plot::
   :include-source: True
   :width: 700

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

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.05,
           "mean_offset": 0.01,
           "outlier_frac": 0.03,
           "outlier_scatter_scale": 0.20,
           "outlier_mean_offset": 0.05,
       },
       "normalize_bins": True,
   }

   tomo = NZTomography()
   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   stats = tomo.cross_bin_stats(
       overlap={"method": "min", "unit": "fraction", "normalize": True, "decimal_places": 6},
       pearson={"normalize": True, "decimal_places": 6},
   )

   overlap = stats["overlap"]
   pearson = stats["pearson"]

   keys = sorted(overlap.keys())

   x_vals = []
   y_vals = []
   labels = []

   for a, i in enumerate(keys):
       for j in keys[a + 1 :]:
           x_vals.append(overlap[i][j])
           y_vals.append(pearson[i][j])
           labels.append(f"({i}, {j})")

   plt.figure(figsize=(7.0, 5.2))
   plt.scatter(x_vals, y_vals, s=80)

   for x, y, label in zip(x_vals, y_vals, labels, strict=True):
       plt.text(x, y, label, fontsize=10, ha="left", va="bottom")

   plt.xlabel("Min overlap [fraction]", fontsize=13)
   plt.ylabel("Pearson correlation", fontsize=13)
   plt.title("Pearson correlation versus overlap", fontsize=14)
   plt.tight_layout()

If the points lie close to a monotonic trend, the two metrics are giving
a broadly consistent picture of pairwise similarity. Deviations from
such a trend indicate bin pairs whose relationship depends more strongly
on whether one emphasizes shared support or overall shape similarity.


Notes
-----

- **Shape statistics** summarize the internal structure of each bin
  curve, such as its center, width, skewness, or peak structure. They
  are useful for understanding how individual bins sample the parent
  redshift distribution.
- **Population statistics** summarize how the galaxy sample is divided
  across bins. These quantities require tomography metadata and
  therefore require rebuilding with ``include_tomo_metadata=True``.
- **Cross-bin diagnostics** summarize how strongly bins are coupled to
  one another. Large overlap, large off-diagonal leakage, or strong
  off-diagonal correlations generally indicate weaker practical
  separation between tomographic bins.
- Equipopulated and equidistant binning can lead to noticeably different
  population balances and cross-bin coupling patterns, even when they
  are constructed from the same parent distribution and photo-z
  uncertainty model.
- The diagnostics returned by :class:`binny.NZTomography` are ordinary
  Python dictionaries, so the quantities shown here can be inspected,
  saved, or reused directly in downstream analysis workflows.