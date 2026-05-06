.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin diagnostics
======================

Once tomographic bins have been built, it is often useful to inspect
their statistical properties before using them in a forecast or analysis.

Binny provides several diagnostics through :class:`binny.NZTomography`
that quantify several complementary aspects of a tomographic binning
scheme: the **shape of the bin curves**, the **distribution of galaxies
across bins**, and the **degree of coupling within or between bin
families** through overlap, leakage, interval-based mass transfer, or
correlated structure. Together, these diagnostics help assess whether
the resulting bins are well separated, well populated, and suitable for
downstream forecasting or cosmological analysis.

This page illustrates these diagnostics for a simple four-bin
photometric example and compares two common binning schemes:

- **equipopulated binning**, where each bin contains a similar fraction
  of the galaxy sample,
- **equidistant binning**, where the redshift interval is divided into
  bins of similar width.

The examples below focus on three parts of the diagnostic workflow:

- visual inspection of the binned redshift curves,
- within-sample diagnostics, such as overlap, leakage, pair rankings,
  and Pearson correlation between bins from the same sample,
- between-sample diagnostics, such as overlap, interval-mass
  summaries, pair rankings, and Pearson correlation between bins from
  different samples.

Together, these quantities provide a practical way to inspect both the
internal structure of a single tomographic sample and the relationship
between two different tomographic samples, for example lens and source
bins in galaxy-galaxy lensing.

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
   :include-source: False
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


Within-sample diagnostics
-------------------------

These diagnostics describe how bins from a single tomographic sample
relate to one another. They are useful for assessing practical bin
separation, internal mixing, and pairwise similarity within one
tomographic realization.


Core cross-bin matrices
~~~~~~~~~~~~~~~~~~~~~~~

Per-bin summaries describe each tomographic bin on its own, but they do
not show how strongly different bins mix with one another. For that, it
is useful to look at **cross-bin diagnostics**.

Binny provides three simple diagnostics for this purpose:

- **overlap**, which measures how much two bin curves lie on top of one
  another in redshift,
- **leakage**, which measures how much of one bin falls inside the
  nominal redshift range of another bin,
- **Pearson correlation**, which measures how similar two bin curves are
  when sampled on the same redshift grid.

Although these quantities are related, they answer slightly different
questions.

**Overlap** is the most direct measure of shared support. If two bins
occupy much of the same redshift range, their overlap will be large. If
they are well separated, their overlap will be small.

**Leakage** is more directional. It asks how much of the content of a
given bin falls inside the intended redshift interval of another bin.
This is useful because a bin can spill into its neighbor more strongly
in one direction than the other, especially when the bin shapes are
asymmetric or when outliers are present.

**Pearson correlation** focuses on shape similarity rather than on
physical bin assignment. Two bins can have a similar overall profile and
therefore a high Pearson correlation even if they are centered at
different redshifts or have different total mass inside their nominal
ranges. Pearson values lie between **-1 and 1**, where **1** indicates
perfect positive correlation, **0** indicates no correlation, and
**-1** indicates perfect negative correlation.

The figure below shows these three diagnostics for a simple four-bin
equidistant photo-z example.

In all three matrices, the diagonal corresponds to comparing a bin with
itself, so those entries are usually the largest or among the largest.
The more interesting part is the off-diagonal structure, because it
shows how strongly different bins are coupled.

A simple way to read the matrices is:

- **small off-diagonal values** usually mean the bins are well separated,
- **large off-diagonal overlap** means two bins share a noticeable part
  of the same redshift range,
- **large off-diagonal leakage** means galaxies assigned to one bin are
  spilling into the nominal interval of another,
- **large off-diagonal Pearson correlation** means two bins have similar
  shapes, even if they do not represent exactly the same redshift slice.

None of these diagnostics by itself determines whether a binning scheme
is “good” or “bad”. Instead, they help reveal different kinds of bin
coupling. In practice, cleaner tomographic binning usually shows a
strong diagonal and weaker off-diagonal structure, while broader
smearing, stronger outliers, or less well-separated bins tend to produce
more visible off-diagonal patterns.


.. plot::
   :include-source: False
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
   result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   bin_edges = result.tomo_meta["bins"]["bin_edges"]

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
       (
           overlap_keys,
           overlap_matrix,
           "Overlap matrix",
           "Tomographic bin",
           "Tomographic bin"
       ),
       (
           leakage_keys,
           leakage_matrix,
           "Leakage matrix",
           "Nominal interval",
           "Input bin"
       ),
       (
           pearson_keys,
           pearson_matrix,
           "Pearson matrix",
           "Tomographic bin",
           "Tomographic bin",
       ),
   ]

   for ax, (keys, matrix, title, xlabel, ylabel) in zip(axes, matrices, strict=True):
       n_rows, n_cols = matrix.shape

       ax.imshow(
           matrix,
           origin="lower",
           aspect="equal",
           cmap="viridis",
           alpha=0.6,
           interpolation="none",
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
       ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
       ax.tick_params(which="minor", bottom=False, left=False)

       for i in range(n_rows):
           for j in range(n_cols):
               ax.text(
                   j,
                   i,
                   f"{matrix[i, j]:.1f}",
                   ha="center",
                   va="center",
                   fontsize=15,
                   color="k",
               )

   plt.tight_layout()

In all three matrices, values near the diagonal typically indicate the
strongest self-association, while off-diagonal structure reveals
coupling between different bins. Large off-diagonal overlap or leakage
usually signals reduced bin separation, whereas large off-diagonal
Pearson correlation indicates that two bins have similar shapes even if
their normalization or interpretation differs.


Comparing multiple similarity metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Different cross-bin metrics highlight different aspects of similarity or
separation between tomographic bins. For that reason, it is often useful
to compare several metrics side by side rather than relying on only one.

Binny provides a range of pairwise diagnostics for this purpose. In the
figure below, the same tomographic bin set is compared using five common
measures:

- **min overlap**, which measures the shared support between two
  normalized bin curves,
- **cosine similarity**, which measures how closely aligned two sampled
  bin vectors are,
- **Jensen--Shannon distance**, which measures how different two
  normalized distributions are,
- **Hellinger distance**, which measures geometric dissimilarity between
  two probability distributions,
- **total variation distance**, which measures the overall discrepancy
  between two normalized distributions.

These metrics do not all follow the same convention.

**Min overlap** and **cosine similarity** are similarity measures, so
**larger values** mean that two bins are more alike.

By contrast, **Jensen--Shannon distance**, **Hellinger distance**, and
**total variation distance** are distance measures, so **smaller
values** mean that two bins are more alike.

This means the matrices should be read in two slightly different ways:

- for **similarity** measures, larger off-diagonal values indicate more
  similar or less well-separated bins,
- for **distance** measures, smaller off-diagonal values indicate more
  similar bins, while larger values indicate stronger separation.

Looking at several metrics together is useful because two bin pairs can
appear similar under one definition and less similar under another. A
pair may share substantial support in redshift, for example, while still
differing in detailed shape, or it may have similar overall structure
without strongly overlapping in the regions that matter most for another
metric.

The diagonal entries again correspond to comparing each bin with itself.
For similarity measures, those entries are typically among the largest.
For distance measures, they are typically **0** or very close to **0**.

The figure below therefore provides a broader view of bin coupling than
any single metric alone. Strong agreement across several metrics usually
indicates a robust pattern, whereas differences between metrics show
that the apparent closeness of two bins depends on how similarity is
being defined.


.. plot::
   :include-source: False
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
       ("Min overlap [%]", overlap_min, "{:.1f}"),
       ("Cosine similarity [%]", overlap_cosine, "{:.1f}"),
       ("JS distance", overlap_js, "{:.2f}"),
       ("Hellinger distance", overlap_hellinger, "{:.2f}"),
       ("TV distance", overlap_tv, "{:.2f}"),
   ]

   fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
   axes = axes.ravel()

   for ax, (title, metric_dict, fmt) in zip(axes[:-1], metric_specs, strict=True):
       keys, matrix = nested_dict_to_matrix(metric_dict)
       n_rows, n_cols = matrix.shape

       ax.imshow(
           matrix,
           origin="lower",
           aspect="equal",
           cmap="viridis",
           alpha=0.6,
           interpolation="none",
       )

       ax.set_title(title)
       ax.set_xticks(np.arange(n_cols))
       ax.set_yticks(np.arange(n_rows))
       ax.set_xticklabels([f"{key + 1}" for key in keys])
       ax.set_yticklabels([f"{key + 1}" for key in keys])
       ax.set_xlabel("Tomographic bin")
       ax.set_ylabel("Tomographic bin")

       ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
       ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
       ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
       ax.tick_params(which="minor", bottom=False, left=False)

       for i in range(n_rows):
           for j in range(n_cols):
               ax.text(
                   j,
                   i,
                   fmt.format(matrix[i, j]),
                   ha="center",
                   va="center",
                   fontsize=13,
                   color="k",
               )

   axes[-1].axis("off")

   plt.tight_layout()


Pair-level summaries
~~~~~~~~~~~~~~~~~~~~

Matrix views are useful for seeing the full coupling pattern at once,
but compact summaries can make the main trends easier to interpret.
The following diagnostics reduce the pairwise information into rankings
or per-bin summaries that are often easier to compare visually.


Ranking the most overlapping bin pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

   labels = [f"({i+1} – {j+1})" for i, j, _ in pair_list]
   values = np.array([value for _, _, value in pair_list])
   y = np.arange(len(labels))

   colors = cmr.take_cmap_colors(
       "viridis",
       len(labels),
       cmap_range=(0.0, 1.0),
       return_fmt="hex",
   )
   fill_colors = [to_rgba(color, 0.6) for color in colors]

   fig, ax = plt.subplots(figsize=(7.8, 4.8))

   ax.barh(
       y,
       values,
       color=fill_colors,
       edgecolor="k",
       linewidth=2.5,
   )

   ax.set_yticks(y)
   ax.set_yticklabels(labels)
   ax.set_xlabel("Min overlap [%]")
   ax.set_ylabel("Bin pair")
   ax.set_title("Ranking of overlapping bin pairs")

   ax.invert_yaxis()

   plt.tight_layout()


Completeness and contamination per bin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The leakage matrix can be summarized into per-bin **completeness** and
**contamination** measures.

For a given input bin, the diagonal leakage entry gives the fraction of
its weight that remains inside its own nominal redshift interval. This
is the **completeness** of that bin. Its complement measures the fraction
that falls outside the intended interval and therefore quantifies
**contamination** by leakage into other nominal bins.

Large completeness indicates that a bin remains well confined to its
target redshift range, while large contamination indicates stronger
mixing with neighboring bins.


.. plot::
   :include-source: False
   :width: 760

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
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
   result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   bin_edges = result.tomo_meta["bins"]["bin_edges"]

   leakage = tomo.cross_bin_stats(
       leakage={"bin_edges": bin_edges, "unit": "percent", "decimal_places": 3},
   )["leakage"]

   keys = sorted(leakage.keys())

   completeness = [leakage[key][key] for key in keys]
   contamination = [100.0 - leakage[key][key] for key in keys]

   x = np.arange(len(keys))
   width = 0.38

   colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.85),
       return_fmt="hex",
   )

   c_comp, c_cont = colors[1], colors[3]

   fill_comp = to_rgba(c_comp, 0.6)
   fill_cont = to_rgba(c_cont, 0.6)

   fig, ax = plt.subplots(figsize=(7.8, 4.8))

   ax.bar(
       x - width / 2,
       completeness,
       width=width,
       color=fill_comp,
       edgecolor="k",
       linewidth=2.5,
       label="Completeness",
   )

   ax.bar(
       x + width / 2,
       contamination,
       width=width,
       color=fill_cont,
       edgecolor="k",
       linewidth=2.5,
       label="Contamination",
   )

   ax.set_xticks(x)
   ax.set_xticklabels([f"Bin {key + 1}" for key in keys])

   ax.set_ylabel("[%]")
   ax.set_xlabel("Tomographic bin")
   ax.set_title("Leakage-based completeness and contamination")

   ax.legend(frameon=False)

   plt.tight_layout()


Leakage composition by nominal bin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A stacked leakage view shows how the weight of each input bin is
distributed across the nominal redshift intervals.

Each bar corresponds to one input bin. The stacked segments show what
fraction of that bin lies inside each target nominal interval. The
diagonal component therefore represents self-consistency, while the
off-diagonal components reveal where leaked weight ends up.

This visualization is especially useful for identifying whether leakage
is mostly local, for example into adjacent bins, or whether there are
longer-range tails and outlier-driven transfers across more distant bins.


.. plot::
   :include-source: False
   :width: 820

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
   result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=photoz_spec,
       include_tomo_metadata=True,
   )

   bin_edges = result.tomo_meta["bins"]["bin_edges"]

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
   fill_colors = [to_rgba(color, 0.6) for color in colors]

   bottoms = np.zeros(len(keys))

   fig, ax = plt.subplots(figsize=(8.2, 5.0))

   for fill_color, target_key in zip(fill_colors, keys, strict=True):
       values = [leakage[source_key][target_key] for source_key in keys]
       ax.bar(
           x,
           values,
           bottom=bottoms,
           color=fill_color,
           edgecolor="k",
           linewidth=2.0,
           label=f"Nominal bin {target_key + 1}",
       )
       bottoms += np.array(values)

   ax.set_xticks(x)
   ax.set_xticklabels([f"Input bin {key + 1}" for key in keys])
   ax.set_xlabel("Input bin")
   ax.set_ylabel("Input-bin mass [%]")
   ax.set_title("Leakage composition")
   ax.legend(frameon=True, loc="center left")

   plt.tight_layout()


Pearson correlation versus overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different pairwise metrics can highlight different aspects of similarity
between tomographic bins.

Here, **min overlap** measures how much two bins directly share support
in redshift, whereas **Pearson correlation** measures how similarly the
two sampled bin curves vary across the redshift grid. Two bins can
therefore exhibit modest direct overlap while still having a relatively
strong shape correlation, or vice versa.

The figure below compares these two metrics for all unique off-diagonal
bin pairs. Each pair of bars corresponds to one bin combination, showing
its min-overlap value and its Pearson correlation side by side.

Because the two metrics capture different aspects of similarity, their
values need not track each other exactly. Some bin pairs may share a
substantial fraction of their redshift support, leading to relatively
large overlap, while still differing in detailed shape. Other pairs may
show similar overall profiles across the redshift grid, producing a
larger Pearson correlation even when their direct overlap is smaller.

Comparing the two metrics in this way helps illustrate how different
definitions of bin similarity emphasize different features of the
tomographic bin curves.


.. plot::
   :include-source: False
   :width: 760

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr
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

   overlap_vals = []
   pearson_vals = []
   labels = []

   for a, i in enumerate(keys):
       for j in keys[a + 1 :]:
           overlap_vals.append(overlap[i][j])
           pearson_vals.append(pearson[i][j])
           labels.append(f"({i+1}–{j+1})")

   overlap_vals = np.array(overlap_vals)
   pearson_vals = np.array(pearson_vals)

   x = np.arange(len(labels))
   width = 0.38

   colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   c_overlap = to_rgba(colors[1], 0.6)
   c_pearson = to_rgba(colors[3], 0.6)

   fig, ax = plt.subplots(figsize=(7.4, 5.2))

   ax.bar(
       x - width / 2,
       overlap_vals,
       width,
       color=c_overlap,
       edgecolor="k",
       linewidth=2.0,
       label="Min overlap",
   )

   ax.bar(
       x + width / 2,
       pearson_vals,
       width,
       color=c_pearson,
       edgecolor="k",
       linewidth=2.0,
       label="Pearson correlation",
   )

   ax.set_xticks(x)
   ax.set_xticklabels(labels)

   ax.set_xlabel("Bin pair")
   ax.set_ylabel("Metric value")
   ax.set_title("Overlap and Pearson correlation by bin pair")

   ax.legend(frameon=True)

   plt.tight_layout()


Between-sample diagnostics
--------------------------

These diagnostics compare bins from two different tomographic samples
rather than within a single one. This is especially useful in joint
analyses, where one often wants to assess how cleanly two samples are
separated or how strongly particular cross-sample bin pairs are coupled.

In the examples below we adopt a simple convention: **lens bins** define
the reference intervals and appear along the horizontal axis, while
**source bins** appear along the vertical axis. For illustration, the
lens sample uses **equidistant binning** and the source sample uses
**equipopulated binning**.


Cross-sample matrices
~~~~~~~~~~~~~~~~~~~~~

In joint analyses it is often useful to compare tomographic bins between
two different samples rather than only within a single sample. A common
example is comparing **lens bins** and **source bins** in a
galaxy-galaxy lensing setup.

Binny provides :meth:`binny.NZTomography.between_sample_stats` for this
purpose. These diagnostics can be used to quantify how strongly bins
from one sample overlap or correlate with bins from another sample, and
how much of one sample falls inside the nominal redshift intervals of
the other.

The example below constructs a simple lens-like and source-like
photo-z setup on the same redshift grid, then compares them using
between-sample overlap, interval-mass, and Pearson correlation.


.. plot::
   :include-source: False
   :width: 980

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import ListedColormap

   from binny import NZTomography

   def nested_rect_dict_to_matrix(nested_dict):
       row_keys = sorted(nested_dict.keys())
       col_keys = sorted(nested_dict[row_keys[0]].keys())
       matrix = np.array(
           [[nested_dict[row_key][col_key] for col_key in col_keys] for row_key in row_keys],
           dtype=float,
       )
       return row_keys, col_keys, matrix

   z = np.linspace(0.0, 2.5, 600)

   lens_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.18,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   source_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.32,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   lens_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.03,
           "mean_offset": 0.00,
           "outlier_frac": 0.01,
           "outlier_scatter_scale": 0.10,
           "outlier_mean_offset": 0.03,
       },
       "normalize_bins": True,
   }

   source_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": 0.06,
           "mean_offset": 0.01,
           "outlier_frac": 0.04,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
       },
       "normalize_bins": True,
   }

   lens = NZTomography()
   lens_result = lens.build_bins(
       z=z,
       nz=lens_nz,
       tomo_spec=lens_spec,
       include_tomo_metadata=True,
   )

   source = NZTomography()
   source.build_bins(
       z=z,
       nz=source_nz,
       tomo_spec=source_spec,
       include_tomo_metadata=True,
   )

   target_edges = lens_result.tomo_meta["bins"]["bin_edges"]

   stats = lens.between_sample_stats(
       source,
       overlap={"method": "min", "unit": "percent", "normalize": True, "decimal_places": 3},
       interval_mass={"target_edges": target_edges, "unit": "percent", "decimal_places": 3},
       pearson={"normalize": True, "decimal_places": 3},
   )

   overlap_rows, overlap_cols, overlap_matrix = nested_rect_dict_to_matrix(stats["overlap"])
   interval_rows, interval_cols, interval_matrix = nested_rect_dict_to_matrix(stats["interval_mass"])
   pearson_rows, pearson_cols, pearson_matrix = nested_rect_dict_to_matrix(stats["pearson"])

   base = plt.get_cmap("viridis")
   colors = base(np.linspace(0.05, 0.95, 256))
   colors[:, -1] = 0.6
   cmap_transparent = ListedColormap(colors)

   fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

   matrices = [
       (
           overlap_rows,
           overlap_cols,
           overlap_matrix,
           "Between-sample overlap",
           "Source bin",
           "Lens bin",
           "{:.1f}",
       ),
       (
           interval_rows,
           interval_cols,
           interval_matrix,
           "Source mass in lens intervals",
           "Lens nominal interval",
           "Source bin",
           "{:.1f}",
       ),
       (
           pearson_rows,
           pearson_cols,
           pearson_matrix,
           "Between-sample Pearson",
           "Source bin",
           "Lens bin",
           "{:.2f}",
       ),
   ]

   for ax, (row_keys, col_keys, matrix, title, xlabel, ylabel, fmt) in zip(
       axes,
       matrices,
       strict=True,
   ):
       n_rows, n_cols = matrix.shape

       ax.imshow(
           matrix,
           origin="lower",
           aspect="auto",
           cmap=cmap_transparent,
           interpolation="none",
       )

       ax.set_title(title)
       ax.set_xticks(np.arange(n_cols))
       ax.set_yticks(np.arange(n_rows))
       ax.set_xticklabels([f"{key + 1}" for key in col_keys])
       ax.set_yticklabels([f"{key + 1}" for key in row_keys])
       ax.set_xlabel(xlabel)
       ax.set_ylabel(ylabel)

       ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
       ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
       ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
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

   plt.tight_layout()

These matrices are generally rectangular rather than square because the
two samples need not have the same binning scheme, bin edges, or parent
redshift distribution. The only requirement is that both tomographic
realizations are evaluated on the same redshift grid. In practice, large
values usually identify lens-source bin combinations that are less cleanly
separated in redshift and may therefore deserve closer inspection in a
joint analysis.


Cross-sample interval-mass composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the interval-mass matrix gives the full rectangular pattern of how
one sample maps onto the nominal intervals of another, a stacked
composition view can make the main structure easier to read at a glance.

In the figure below, each bar corresponds to one **source bin**. The
stacked segments show what fraction of that source-bin mass falls inside
each **lens nominal interval**. The full height of a bar therefore
represents the total source-bin mass accounted for across the chosen
lens intervals, while the relative segment sizes show how that mass is
distributed.

This view is useful for diagnosing whether a source bin is associated
primarily with a single lens interval or whether its weight is spread
more broadly across several intervals. A source bin whose bar is
dominated by one segment is more cleanly aligned with a particular lens
interval, whereas a bar with several substantial segments indicates
broader cross-sample mixing.

In many practical cases, neighboring intervals receive the largest
secondary contributions, reflecting partial redshift overlap between the
two samples. More extended tails across distant intervals can instead
indicate broader smearing or stronger outlier-driven transfer.

.. plot::
   :include-source: False
   :width: 820

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import NZTomography

   z = np.linspace(0.0, 2.5, 600)

   lens_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.18,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   source_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.32,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   lens_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.03,
           "mean_offset": 0.00,
           "outlier_frac": 0.01,
           "outlier_scatter_scale": 0.10,
           "outlier_mean_offset": 0.03,
       },
       "normalize_bins": True,
   }

   source_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": 0.06,
           "mean_offset": 0.01,
           "outlier_frac": 0.04,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
       },
       "normalize_bins": True,
   }

   lens = NZTomography()
   lens_result = lens.build_bins(
       z=z,
       nz=lens_nz,
       tomo_spec=lens_spec,
       include_tomo_metadata=True,
   )

   source = NZTomography()
   source.build_bins(
       z=z,
       nz=source_nz,
       tomo_spec=source_spec,
       include_tomo_metadata=True,
   )

   target_edges = lens_result.tomo_meta["bins"]["bin_edges"]

   interval_mass = lens.between_sample_stats(
       source,
       interval_mass={
           "target_edges": target_edges,
           "unit": "percent",
           "decimal_places": 3,
       },
   )["interval_mass"]

   source_keys = sorted(interval_mass.keys())
   lens_keys = sorted(interval_mass[source_keys[0]].keys())

   x = np.arange(len(source_keys))

   colors = cmr.take_cmap_colors(
       "viridis",
       len(lens_keys),
       cmap_range=(0.1, 0.9),
       return_fmt="hex",
   )
   fill_colors = [to_rgba(color, 0.6) for color in colors]

   bottoms = np.zeros(len(source_keys))

   fig, ax = plt.subplots(figsize=(8.2, 5.0))

   for fill_color, lens_key in zip(fill_colors, lens_keys, strict=True):
       values = np.array(
           [interval_mass[source_key][lens_key] for source_key in source_keys],
           dtype=float,
       )
       ax.bar(
           x,
           values,
           bottom=bottoms,
           color=fill_color,
           edgecolor="k",
           linewidth=2.0,
           label=f"Lens interval {lens_key + 1}",
       )
       bottoms += values

   ax.set_xticks(x)
   ax.set_xticklabels([f"Source bin {key + 1}" for key in source_keys])
   ax.set_xlabel("Source bin")
   ax.set_ylabel("Percent of source-bin mass")
   ax.set_title("Source-bin mass across lens intervals")
   ax.legend(frameon=True, loc= "center left")

   plt.tight_layout()


A bar dominated by one segment indicates that most of the corresponding
source-bin mass falls inside a single lens interval, suggesting cleaner
cross-sample alignment. By contrast, bars with several sizable segments
show that the source bin is distributed more broadly across the lens
intervals and is therefore less cleanly separated in redshift.

As in the matrix view, the most informative structure is usually in the
off-dominant components. Small secondary segments suggest modest
spillover into neighboring lens intervals, whereas larger secondary
contributions indicate stronger cross-sample mixing.


Cross-sample pair rankings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Between-sample pair rankings provide a compact way to identify which bin
combinations across two tomographic samples are most strongly coupled
according to a chosen metric.

This is often useful in joint analyses, for example when identifying
which lens-source bin combinations are most strongly overlapping in
redshift. Such rankings can help diagnose where sample separation is
cleanest and where additional care may be needed in downstream analysis.

The example below ranks lens-source bin pairs by their between-sample
min-overlap score.


.. plot::
   :include-source: False
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import NZTomography

   z = np.linspace(0.0, 2.5, 600)

   lens_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.18,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   source_nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.32,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   lens_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": 0.03,
           "mean_offset": 0.00,
           "outlier_frac": 0.01,
           "outlier_scatter_scale": 0.10,
           "outlier_mean_offset": 0.03,
       },
       "normalize_bins": True,
   }

   source_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": 0.06,
           "mean_offset": 0.01,
           "outlier_frac": 0.04,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
       },
       "normalize_bins": True,
   }

   lens = NZTomography()
   lens.build_bins(
       z=z,
       nz=lens_nz,
       tomo_spec=lens_spec,
       include_tomo_metadata=True,
   )

   source = NZTomography()
   source.build_bins(
       z=z,
       nz=source_nz,
       tomo_spec=source_spec,
       include_tomo_metadata=True,
   )

   pair_list = lens.between_sample_stats(
       source,
       pairs={
           "method": "min",
           "unit": "percent",
           "threshold": 0.0,
           "direction": "high",
           "normalize": True,
           "decimal_places": 3,
       },
   )["correlations"]

   labels = [f"L{i+1} - S{j+1}" for i, j, _ in pair_list]
   values = np.array([value for _, _, value in pair_list])
   y = np.arange(len(labels))

   colors = cmr.take_cmap_colors(
       "viridis",
       len(labels),
       cmap_range=(0.0, 1.0),
       return_fmt="hex",
   )
   fill_colors = [to_rgba(color, 0.6) for color in colors]

   fig, ax = plt.subplots(figsize=(8.2, 7))

   ax.barh(
       y,
       values,
       color=fill_colors,
       edgecolor="k",
       linewidth=2.5,
   )

   ax.set_yticks(y)
   ax.set_yticklabels(labels)
   ax.set_xlabel("Between-sample min overlap [%]")
   ax.set_ylabel("Lens–source pair")
   ax.set_title("Ranking of cross-sample overlapping pairs")

   ax.invert_yaxis()

   plt.tight_layout()


Notes
-----

- **Within-sample diagnostics** summarize how strongly bins from the
  same tomographic sample are coupled to one another. Large overlap,
  large off-diagonal leakage, or strong off-diagonal correlations
  generally indicate weaker practical separation between tomographic
  bins.
- **Between-sample diagnostics** compare bins from two different
  tomographic samples, such as lens and source populations. They are
  useful for assessing redshift separation, cross-sample similarity,
  interval-based overlap, and possible foreground contamination in joint
  analyses.
- Equipopulated and equidistant binning can lead to noticeably different
  population balances and coupling patterns, even when they are
  constructed from the same parent distribution and photo-z uncertainty
  model.
- The diagnostics returned by :class:`binny.NZTomography` are ordinary
  Python dictionaries, so the quantities shown here can be inspected,
  saved, or reused directly in downstream analysis workflows.