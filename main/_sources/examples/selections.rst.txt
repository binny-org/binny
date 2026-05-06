.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin-pair selections
==========================

Tomographic analyses often construct many possible combinations of bins,
but only a subset of those combinations are typically used in a downstream
analysis.

For example, one may

- exclude auto-correlations,
- require bins to appear in a consistent redshift order,
- remove pairs whose redshift support overlaps too strongly,
- select only well-separated lens–source combinations.

In Binny, such rules are expressed through a **selection specification**
applied to tomographic bins using

:meth:`binny.NZTomography.bin_combo_filter`.

Selections combine two ideas:

- a **topology**, which defines the initial set of candidate tuples,
- one or more **filters**, which remove tuples that do not satisfy
  a chosen criterion.

By convention, Binny assumes an **upper-triangular ordering**
:math:`i \le j` when constructing pairs within a single sample.
This avoids symmetric duplicates such as :math:`(i,j)` and :math:`(j,i)`.
The reasoning behind this convention is discussed in
:doc:`../theory/conventions`.

The examples below illustrate the most common patterns.


Basic workflow
--------------

Selections are written as a dictionary describing the topology and any
filters to apply.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
       "filters": [
           {"name": "score_relation", "score": "peak", "relation": "lt"},
       ],
   }

   selected_pairs = tomo.bin_combo_filter(spec)

This returns a list of tuples such as

.. code-block:: python

   [(0, 1), (0, 2), (1, 2)]

which can then be used in a downstream analysis.


Example setup
-------------

The examples below assume that a tomography object has already been built.

.. code-block:: python

   import numpy as np
   from binny import NZTomography

   z = np.linspace(0.0, 3.0, 500)
   nz = NZTomography.nz_model("smail", z, alpha=2.0, beta=1.5, z0=0.5)

   tomo = NZTomography()

   tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec={
           "kind": "photoz",
           "bins": {
               "scheme": "equipopulated",
               "n_bins": 4,
           },
       },
   )

The selections below operate on the cached bin curves stored in
``tomo``.


Selections based on bin statistics
----------------------------------

Some selection rules rely only on **summary statistics of individual
bin curves**, such as their peak location, mean redshift, or credible
width.

These statistics describe properties of each bin independently and are
often used to enforce ordering or separation conditions between bins.


Generating candidate bin pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before applying filters, a topology defines which tuples are considered.

For example, to construct all ordered bin pairs:

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
   }

   pairs = tomo.bin_combo_filter(spec)

For four bins this would produce

.. code-block:: python

   [
       (0,0), (0,1), (0,2), (0,3),
       (1,0), (1,1), (1,2), (1,3),
       (2,0), (2,1), (2,2), (2,3),
       (3,0), (3,1), (3,2), (3,3),
   ]

A common alternative is to keep only the **upper triangle**
:math:`i \le j` to avoid symmetric duplicates.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
   }

   pairs = tomo.bin_combo_filter(spec)

For four bins this would produce

.. code-block:: python

   [
       (0,0), (0,1), (0,2), (0,3),
              (1,1), (1,2), (1,3),
                     (2,2), (2,3),
                            (3,3),
   ]

Requiring redshift ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple selection rule is to require that the second bin peaks at a
higher redshift than the first.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "score_relation",
               "score": "peak",
               "pos_a": 0,
               "pos_b": 1,
               "relation": "gt",
           }
       ],
   }

   ordered_pairs = tomo.bin_combo_filter(spec)

This keeps only pairs satisfying

.. math::

   z_{\mathrm{peak}}(j) > z_{\mathrm{peak}}(i).

Such ordering rules are often useful when enforcing a physically
meaningful redshift hierarchy.


Requiring minimum separation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two bins may be too similar if their effective redshift locations are
very close.

A minimum separation can be imposed using a score summary such as the
peak or mean redshift.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
       "filters": [
           {
               "name": "score_separation",
               "score": "peak",
               "min_sep": 0.2,
               "absolute": True,
           }
       ],
   }

   separated_pairs = tomo.bin_combo_filter(spec)

This keeps only pairs whose peak locations differ by at least ``0.2``.


Comparing bin widths
^^^^^^^^^^^^^^^^^^^^

Bin widths describe how broad each bin is in redshift. Some analyses
prefer to compare bins of similar width.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_off_diagonal"},
       "filters": [
           {
               "name": "width_ratio",
               "max_ratio": 1.5,
               "symmetric": True,
               "mass": 0.68,
           }
       ],
   }

   compatible_pairs = tomo.bin_combo_filter(spec)

This keeps only pairs whose credible-width ratio does not exceed ``1.5``.


Available statistic-based filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following filters operate on **summary statistics computed for each
individual bin curve**. These statistics summarize properties such as
the location or width of a bin in redshift space.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30
   :class: binny-table

   * - Filter name
     - Description
     - Typical use
   * - ``score_relation``
     - Compares a chosen summary statistic between two bins using a
       relation such as greater-than or less-than.
     - Enforcing redshift ordering between bins.
   * - ``score_separation``
     - Requires the difference between two bin statistics to exceed a
       specified minimum separation.
     - Avoiding pairs of bins that are too close in redshift.
   * - ``score_difference``
     - Filters pairs based on the signed or absolute difference between
       two summary statistics.
     - Selecting pairs with a specific spacing pattern.
   * - ``score_consistency``
     - Requires two summary statistics to satisfy a consistency relation
       across bins.
     - Enforcing monotonic ordering or alignment of statistics.
   * - ``width_ratio``
     - Compares the credible widths of two bins and requires their ratio
       to remain below a specified threshold.
     - Selecting bins with similar redshift widths.

The statistics used by these filters include quantities such as

- peak location,
- mean redshift,
- median redshift,
- credible width.

Because these filters operate on **per-bin statistics**, they depend only
on properties of individual bins rather than the detailed overlap
between curves.


Selections driven by diagnostics
--------------------------------

Selections can also be motivated by **diagnostic quantities that compare
two bin curves directly**.

Examples include overlap fractions, similarity metrics, or integrated
curve norms.

Such diagnostics help reveal whether bins are strongly coupled or
effectively redundant.


Removing strongly overlapping pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common diagnostic is the **overlap fraction** between two bin curves.

Pairs with large overlap may contain highly redundant information and
can therefore be excluded.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_off_diagonal"},
       "filters": [
           {
               "name": "overlap_fraction",
               "threshold": 0.25,
               "compare": "le",
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

This keeps only pairs whose overlap fraction does not exceed ``0.25``.


Using normalized overlap measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another common diagnostic is the **overlap coefficient**, which measures
how strongly one curve is contained within another.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "overlap_coefficient",
               "threshold": 0.4,
               "compare": "le",
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)


Filtering using custom metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users may define their own diagnostic metric.

.. code-block:: python

   import numpy as np
   import binny

   def l1_distance(c1, c2):
       return float(np.trapezoid(np.abs(c1 - c2)))

   binny.register_metric_kernel("l1_distance", l1_distance)

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "metric",
               "metric": "l1_distance",
               "threshold": 0.2,
               "compare": "ge",
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

This allows arbitrary similarity or distance measures to be used when
selecting bin combinations.


Available diagnostic-based filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Diagnostic-based filters operate directly on **pairs of bin curves**.
Instead of using summary statistics, these filters evaluate quantities
that measure how similar or strongly coupled two curves are.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30
   :class: binny-table

   * - Filter name
     - Description
     - Typical use
   * - ``overlap_fraction``
     - Measures the fractional overlap between two bin curves based on
       their integrated support.
     - Removing bins that share too much redshift support.
   * - ``overlap_coefficient``
     - Computes the normalized overlap coefficient, which measures how
       strongly one curve is contained within another.
     - Identifying redundant bins or nested redshift distributions.
   * - ``metric``
     - Applies a user-defined metric to the pair of curves and filters
       pairs according to a threshold.
     - Using custom similarity or distance measures.
   * - ``curve_norm_threshold``
     - Filters pairs based on the magnitude of a curve-based norm
       computed from the two curves.
     - Removing pairs with insufficient signal or excessively large
       combined amplitude.

These diagnostics are useful when selection criteria depend on the
**detailed shapes of bin curves**, rather than on simple summary
statistics.


Visual examples
---------------

The sections above illustrate how selections are defined and applied in
code.

The next section presents **visual examples** showing how diagnostic
matrices and candidate pair grids can be used to understand and
illustrate bin-combination selections.

Lens--source overlap filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example illustrates a diagnostic-driven workflow for
**between-sample pair selection**, which commonly appears in analyses
such as galaxy-galaxy lensing.

The top panel shows the **lens and source tomographic bins** in
redshift space. Lens bins are displayed with dashed hatched curves,
while source bins are shown as solid filled curves.

The first matrix shows the **candidate lens-source topology**,
constructed as the full Cartesian product of lens and source bins.
Each cell represents a possible pair of lens bin :math:`i` and source
bin :math:`j`.

The middle matrix shows the **normalized min-overlap diagnostic**
between the two samples. Each entry measures the fraction of overlap
between the corresponding lens and source bin curves. Larger values
indicate stronger coupling between the two bins.

The final matrix shows the **result of applying an overlap threshold**.
Pairs whose overlap exceeds the chosen limit are marked as excluded,
leaving only the combinations that satisfy the diagnostic criterion.


.. plot::
   :include-source: False
   :width: 1100

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.patches import Polygon, Rectangle

   from binny import NZTomography


   def plot_two_samples(
       ax,
       z,
       lens_bins,
       source_bins,
       title,
       lens_cmap="viridis",
       source_cmap="viridis",
       lens_cmap_range=(0.10, 0.80),
       source_cmap_range=(0.20, 1.00),
   ):
       lens_keys = sorted(lens_bins.keys())
       source_keys = sorted(source_bins.keys())

       lens_colors = cmr.take_cmap_colors(
           lens_cmap,
           len(lens_keys),
           cmap_range=lens_cmap_range,
           return_fmt="hex",
       )
       source_colors = cmr.take_cmap_colors(
           source_cmap,
           len(source_keys),
           cmap_range=source_cmap_range,
           return_fmt="hex",
       )

       # Lenses: hatched + dashed
       for i, (color, key) in enumerate(zip(lens_colors, lens_keys, strict=True)):
           curve = np.asarray(lens_bins[key], dtype=float)
           ax.fill_between(
               z,
               0.0,
               curve,
               facecolor=color,
               alpha=0.65,
               linewidth=0.0,
               hatch="///",
               edgecolor=color,
               zorder=10 + i,
           )
           ax.plot(
               z,
               curve,
               color="k",
               linewidth=1.8,
               linestyle="--",
               zorder=20 + i,
           )

       # Sources: solid filled
       for i, (color, key) in enumerate(zip(source_colors, source_keys, strict=True)):
           curve = np.asarray(source_bins[key], dtype=float)
           ax.fill_between(
               z,
               0.0,
               curve,
               color=color,
               alpha=0.65,
               linewidth=0.0,
               zorder=40 + i,
           )
           ax.plot(
               z,
               curve,
               color="k",
               linewidth=1.8,
               zorder=50 + i,
           )

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")
       ax.set_ylabel(r"Normalized $n_i(z)$")

       ax.text(
           0.98,
           0.96,
           "Hatched dashed: lens bins\nSolid filled: source bins",
           transform=ax.transAxes,
           ha="right",
           va="top",
           bbox=dict(
               boxstyle="round",
               facecolor="white",
               alpha=0.9,
               edgecolor="none",
           ),
       )


   def square_triangles(x, y, size=0.42):
       x0, x1 = x - size, x + size
       y0, y1 = y - size, y + size

       tri1 = [(x0, y0), (x1, y0), (x0, y1)]
       tri2 = [(x1, y1), (x1, y0), (x0, y1)]
       border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
       diag = [(x1, y0), (x0, y1)]

       return tri1, tri2, border, diag


   def draw_pair_cell(ax, row, col, lens_color, source_color, size=0.42):
       tri1, tri2, border, diag = square_triangles(col, row, size=size)

       # Lower-left triangle: lens -> hatched
       ax.add_patch(
           Polygon(
               tri1,
               closed=True,
               facecolor=lens_color,
               edgecolor=lens_color,
               hatch="///",
               linewidth=0.0,
               alpha=0.65,
               zorder=3,
           )
       )

       # Upper-right triangle: source -> solid filled
       ax.add_patch(
           Polygon(
               tri2,
               closed=True,
               facecolor=source_color,
               edgecolor="none",
               alpha=0.65,
               zorder=3,
           )
       )

       ax.add_patch(
           Polygon(
               border,
               closed=True,
               facecolor="none",
               edgecolor="k",
               linewidth=1.8,
               zorder=4,
           )
       )
       ax.plot(
           [diag[0][0], diag[1][0]],
           [diag[0][1], diag[1][1]],
           color="k",
           linewidth=1.2,
           zorder=5,
       )


   def draw_exclusion_overlay(ax, row, col, size=0.42):
       x0 = col - size
       y0 = row - size
       width = 2.0 * size
       height = 2.0 * size

       ax.add_patch(
           Rectangle(
               (x0, y0),
               width,
               height,
               facecolor="0.85",
               edgecolor="k",
               linewidth=1.5,
               alpha=0.65,
               zorder=10,
           )
       )
       ax.plot(
           [x0, x0 + width],
           [y0, y0 + height],
           color="k",
           linewidth=2.0,
           zorder=11,
       )
       ax.plot(
           [x0, x0 + width],
           [y0 + height, y0],
           color="k",
           linewidth=2.0,
           zorder=11,
       )


   def setup_rect_pair_axes(ax, n_rows, n_cols, title):
       ax.set_title(title)
       ax.set_xlim(-0.5, n_cols - 0.5)
       ax.set_ylim(n_rows - 0.5, -0.5)

       ax.set_xticks(range(n_cols))
       ax.set_yticks(range(n_rows))
       ax.set_xticklabels([f"{j + 1}" for j in range(n_cols)])
       ax.set_yticklabels([f"{i + 1}" for i in range(n_rows)])

       ax.set_xlabel("Source bin $j$")
       ax.set_ylabel("Lens bin $i$")

       for k in range(n_rows + 1):
           ax.axhline(k - 0.5, color="k", linewidth=1.0, zorder=1)
       for k in range(n_cols + 1):
           ax.axvline(k - 0.5, color="k", linewidth=1.0, zorder=1)

       for side in ["left", "right", "top", "bottom"]:
           ax.spines[side].set_visible(True)
           ax.spines[side].set_linewidth(1.8)

       ax.tick_params(
           axis="both",
           which="both",
           direction="in",
           top=True,
           right=True,
           width=1.5,
           length=5,
       )

       ax.grid(False)


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
   source_result = source.build_bins(
       z=z,
       nz=source_nz,
       tomo_spec=source_spec,
       include_tomo_metadata=True,
   )

   between_overlap = lens.between_sample_stats(
       source,
       overlap={"method": "min", "unit": "fraction", "normalize": True, "decimal_places": 6},
   )["overlap"]

   lens_keys, source_keys, overlap_matrix = nested_rect_dict_to_matrix(between_overlap)

   n_lens = len(lens_keys)
   n_source = len(source_keys)

   threshold = 0.2

   spec = {
       "topology": {"name": "pairs_cartesian"},
       "filters": [
           {
               "name": "overlap_fraction",
               "threshold": threshold,
               "compare": "le",
           }
       ],
   }

   selected_pairs_raw = lens.bin_combo_filter(spec, other=source)
   selected_pairs = set(tuple(pair) for pair in selected_pairs_raw)

   candidate_pairs = [
       (i_key, j_key)
       for i_key in lens_keys
       for j_key in source_keys
   ]

   excluded_pairs = [
       pair for pair in candidate_pairs
       if pair not in selected_pairs
   ]

   lens_pos = {key: idx for idx, key in enumerate(lens_keys)}
   source_pos = {key: idx for idx, key in enumerate(source_keys)}

   lens_colors = cmr.take_cmap_colors(
       "viridis",
       n_lens,
       cmap_range=(0.10, 0.80),
       return_fmt="hex",
   )

   source_colors = cmr.take_cmap_colors(
       "viridis",
       n_source,
       cmap_range=(0.20, 1.00),
       return_fmt="hex",
   )

   fig = plt.figure(figsize=(15.2, 8.8))
   gs = fig.add_gridspec(
       2, 3, height_ratios=[1.0, 1.0],
       hspace=0.38, wspace=0.28)

   ax_top = fig.add_subplot(gs[0, :])
   ax0 = fig.add_subplot(gs[1, 0])
   ax1 = fig.add_subplot(gs[1, 1])
   ax2 = fig.add_subplot(gs[1, 2])

   # Top panel: lens and source bins in redshift space
   plot_two_samples(
       ax_top,
       z,
       lens_result.bins,
       source_result.bins,
       "Lens and source tomographic bins",
   )

   # Panel 1: candidate Cartesian topology
   setup_rect_pair_axes(ax0, n_lens, n_source, "Candidate lens-source pairs")
   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = source_pos[j_key]
       draw_pair_cell(ax0, i, j, lens_colors[i], source_colors[j])

   # Panel 2: between-sample overlap matrix
   ax1.imshow(
       overlap_matrix,
       origin="upper",
       aspect="auto",
       cmap="viridis",
       alpha=0.65,
       interpolation="none",
   )

   ax1.set_title("Overlap matrix")
   ax1.set_xticks(np.arange(n_source))
   ax1.set_yticks(np.arange(n_lens))
   ax1.set_xticklabels([f"{k + 1}" for k in source_keys])
   ax1.set_yticklabels([f"{k + 1}" for k in lens_keys])
   ax1.set_xlabel("Source bin $j$")
   ax1.set_ylabel("Lens bin $i$")

   ax1.set_xticks(np.arange(-0.5, n_source, 1), minor=True)
   ax1.set_yticks(np.arange(-0.5, n_lens, 1), minor=True)
   ax1.grid(which="minor", color="k", linestyle="-", linewidth=1.2)
   ax1.tick_params(which="minor", bottom=False, left=False)

   for side in ["left", "right", "top", "bottom"]:
       ax1.spines[side].set_visible(True)
       ax1.spines[side].set_linewidth(1.8)

   ax1.tick_params(
       axis="both",
       which="both",
       direction="in",
       top=True,
       right=True,
       width=1.5,
       length=5,
   )

   for i in range(n_lens):
       for j in range(n_source):
           value = overlap_matrix[i, j]
           txt = f"{value:.2f}"
           color = "k" if value > threshold else "white"
           ax1.text(
               j,
               i,
               txt,
               ha="center",
               va="center",
               color=color,
               zorder=5,
           )

   # Panel 3: pairs excluded by BinComboFilter
   setup_rect_pair_axes(
       ax2,
       n_lens,
       n_source,
       rf"Excluded pairs $>{100 * threshold:.0f}\%$ overlap",
   )
   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = source_pos[j_key]
       draw_pair_cell(ax2, i, j, lens_colors[i], source_colors[j])

   for i_key, j_key in excluded_pairs:
       i = lens_pos[i_key]
       j = source_pos[j_key]
       draw_exclusion_overlay(ax2, i, j)

   plt.tight_layout()


Leakage-based filtering
^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates diagnostic-driven filtering for
**within-sample bin pairs**, which arises in analyses such as galaxy
clustering.

The top panel shows the tomographic **lens-bin redshift distributions**.

The first matrix shows the **candidate pair topology**, constructed
from the upper-triangular set of unique bin pairs, including
auto-correlations. The lower triangle is masked to emphasize that only
unique pairs are considered.

The middle matrix shows the **symmetrized leakage diagnostic**. For
each pair of bins, the diagnostic takes the larger of the two
directional leakage values. This measures the degree to which galaxies
from one bin contaminate another.

The final matrix shows the **pairs retained after applying a leakage
threshold**. Off-diagonal pairs with leakage above the chosen limit are
excluded, while auto-correlations are always kept.


.. plot::
   :include-source: False
   :width: 1100

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.patches import Polygon, Rectangle

   from binny import NZTomography


   def plot_one_sample(
       ax,
       z,
       bins,
       title,
       cmap="viridis",
       cmap_range=(0.10, 0.90),
   ):
       keys = sorted(bins.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bins[key], dtype=float)
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")
       ax.set_ylabel(r"Normalized $n_i(z)$")


   def square_triangles(x, y, size=0.42):
       x0, x1 = x - size, x + size
       y0, y1 = y - size, y + size

       tri1 = [(x0, y0), (x1, y0), (x0, y1)]
       tri2 = [(x1, y1), (x1, y0), (x0, y1)]
       border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
       diag = [(x1, y0), (x0, y1)]

       return tri1, tri2, border, diag


   def draw_pair_cell(ax, row, col, color_i, color_j, size=0.42):
       tri1, tri2, border, diag = square_triangles(col, row, size=size)

       ax.add_patch(
           Polygon(
               tri1,
               closed=True,
               facecolor=color_i,
               edgecolor="none",
               alpha=0.65,
               zorder=3,
           )
       )

       ax.add_patch(
           Polygon(
               tri2,
               closed=True,
               facecolor=color_j,
               edgecolor="none",
               alpha=0.65,
               zorder=3,
           )
       )

       ax.add_patch(
           Polygon(
               border,
               closed=True,
               facecolor="none",
               edgecolor="k",
               linewidth=1.8,
               zorder=4,
           )
       )

       ax.plot(
           [diag[0][0], diag[1][0]],
           [diag[0][1], diag[1][1]],
           color="k",
           linewidth=1.2,
           zorder=5,
       )


   def draw_exclusion_overlay(ax, row, col, size=0.42):
       x0 = col - size
       y0 = row - size
       width = 2.0 * size
       height = 2.0 * size

       ax.add_patch(
           Rectangle(
               (x0, y0),
               width,
               height,
               facecolor="0.85",
               edgecolor="k",
               linewidth=1.5,
               alpha=0.65,
               zorder=10,
           )
       )
       ax.plot(
           [x0, x0 + width],
           [y0, y0 + height],
           color="k",
           linewidth=2.0,
           zorder=11,
       )
       ax.plot(
           [x0, x0 + width],
           [y0 + height, y0],
           color="k",
           linewidth=2.0,
           zorder=11,
       )


   def setup_square_pair_axes(ax, n_bins, title):
       ax.set_title(title)
       ax.set_xlim(-0.5, n_bins - 0.5)
       ax.set_ylim(n_bins - 0.5, -0.5)

       ax.set_xticks(range(n_bins))
       ax.set_yticks(range(n_bins))
       ax.set_xticklabels([f"{j + 1}" for j in range(n_bins)])
       ax.set_yticklabels([f"{i + 1}" for i in range(n_bins)])

       ax.set_xlabel("Lens bin $j$")
       ax.set_ylabel("Lens bin $i$")

       for k in range(n_bins + 1):
           ax.axhline(k - 0.5, color="k", linewidth=1.0, zorder=1)
           ax.axvline(k - 0.5, color="k", linewidth=1.0, zorder=1)

       for side in ["left", "right", "top", "bottom"]:
           ax.spines[side].set_visible(True)
           ax.spines[side].set_linewidth(1.8)

       ax.tick_params(
           axis="both",
           which="both",
           direction="in",
           top=True,
           right=True,
           width=1.5,
           length=5,
       )

       ax.grid(False)


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

   lens_spec = {
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

   lens = NZTomography()
   lens_result = lens.build_bins(
       z=z,
       nz=nz,
       tomo_spec=lens_spec,
       include_tomo_metadata=True,
   )

   bin_edges = lens_result.tomo_meta["bins"]["bin_edges"]

   leakage = lens.cross_bin_stats(
       leakage={"bin_edges": bin_edges, "unit": "fraction", "decimal_places": 6},
   )["leakage"]

   lens_keys, leakage_matrix = nested_dict_to_matrix(leakage)
   n_lens = len(lens_keys)

   # Build a symmetric leakage diagnostic by taking the larger
   # of the two directional leakage values for each pair.
   sym_leakage_matrix = np.maximum(leakage_matrix, leakage_matrix.T)

   threshold = 0.12

   candidate_pairs = [
       (i_key, j_key)
       for a, i_key in enumerate(lens_keys)
       for j_key in lens_keys[a:]
   ]

   retained_pairs = []
   excluded_pairs = []

   for i_key, j_key in candidate_pairs:
       i = lens_keys.index(i_key)
       j = lens_keys.index(j_key)

       if i == j:
           retained_pairs.append((i_key, j_key))
       elif sym_leakage_matrix[i, j] <= threshold:
           retained_pairs.append((i_key, j_key))
       else:
           excluded_pairs.append((i_key, j_key))

   lens_pos = {key: idx for idx, key in enumerate(lens_keys)}

   lens_colors = cmr.take_cmap_colors(
       "viridis",
       n_lens,
       cmap_range=(0.10, 0.90),
       return_fmt="hex",
   )

   fig = plt.figure(figsize=(15.2, 8.8))
   gs = fig.add_gridspec(
       2, 3, height_ratios=[1.0, 1.0],
       hspace=0.38, wspace=0.28)

   ax_top = fig.add_subplot(gs[0, :])
   ax0 = fig.add_subplot(gs[1, 0])
   ax1 = fig.add_subplot(gs[1, 1])
   ax2 = fig.add_subplot(gs[1, 2])

   # Top panel: lens bins in redshift space
   plot_one_sample(
       ax_top,
       z,
       lens_result.bins,
       "Lens tomographic bins",
   )

   # Panel 1: candidate symmetric topology
   setup_square_pair_axes(ax0, n_lens, "Candidate pairs")
   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_pair_cell(ax0, i, j, lens_colors[i], lens_colors[j])

   # Mask lower triangle so the unique-pair topology is visually clear
   for i in range(n_lens):
       for j in range(i):
           ax0.add_patch(
               Rectangle(
                   (j - 0.5, i - 0.5),
                   1.0,
                   1.0,
                   facecolor="white",
                   edgecolor="none",
                   zorder=20,
               )
           )

   # Re-draw grid lines on top of the mask
   for k in range(n_lens + 1):
       ax0.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
       ax0.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

   # Panel 2: symmetrized leakage matrix
   ax1.imshow(
       sym_leakage_matrix,
       origin="upper",
       aspect="auto",
       cmap="viridis",
       alpha=0.65,
       interpolation="none",
   )

   ax1.set_title("Leakage matrix")
   ax1.set_xticks(np.arange(n_lens))
   ax1.set_yticks(np.arange(n_lens))
   ax1.set_xticklabels([f"{k + 1}" for k in lens_keys])
   ax1.set_yticklabels([f"{k + 1}" for k in lens_keys])
   ax1.set_xlabel("Lens bin $j$")
   ax1.set_ylabel("Lens bin $i$")

   ax1.set_xticks(np.arange(-0.5, n_lens, 1), minor=True)
   ax1.set_yticks(np.arange(-0.5, n_lens, 1), minor=True)
   ax1.grid(which="minor", color="k", linestyle="-", linewidth=1.2)
   ax1.tick_params(which="minor", bottom=False, left=False)

   for side in ["left", "right", "top", "bottom"]:
       ax1.spines[side].set_visible(True)
       ax1.spines[side].set_linewidth(1.8)

   ax1.tick_params(
       axis="both",
       which="both",
       direction="in",
       top=True,
       right=True,
       width=1.5,
       length=5,
   )

   for i in range(n_lens):
       for j in range(n_lens):
           value = sym_leakage_matrix[i, j]
           txt = f"{value:.2f}"
           color = "k" if value > threshold else "white"
           ax1.text(
               j,
               i,
               txt,
               ha="center",
               va="center",
               color=color,
               zorder=5,
           )

   # Panel 3: retained pairs after leakage cut
   setup_square_pair_axes(
       ax2,
       n_lens,
       ax2_title := rf"Retained pairs $\leq {100 * threshold:.0f}\%$",
   )

   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_pair_cell(ax2, i, j, lens_colors[i], lens_colors[j])

   for i in range(n_lens):
       for j in range(i):
           ax2.add_patch(
               Rectangle(
                   (j - 0.5, i - 0.5),
                   1.0,
                   1.0,
                   facecolor="white",
                   edgecolor="none",
                   zorder=20,
               )
           )

   for i_key, j_key in excluded_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_exclusion_overlay(ax2, i, j)

   for k in range(n_lens + 1):
       ax2.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
       ax2.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

   plt.tight_layout()


Population-fraction filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example illustrates filtering based on **population statistics**
rather than the shapes of the bin curves.

The top panel shows the tomographic **lens-bin redshift
distributions**, along with the fraction of galaxies contained in each
bin.

The first matrix again shows the **candidate set of unique bin pairs**,
including auto-correlations.

The middle matrix shows the **pairwise minimum bin-fraction
diagnostic**. For each pair of bins, the value corresponds to the
smaller of the two galaxy fractions. This highlights pairs that
involve poorly populated bins.

The final matrix shows the **result of applying a minimum population
threshold**. Pairs whose minimum bin fraction falls below the chosen
limit are excluded, leaving only pairs built from sufficiently
populated bins.



.. plot::
   :include-source: False
   :width: 1100

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.patches import Polygon, Rectangle

   from binny import NZTomography


   def plot_one_sample(
       ax,
       z,
       bins,
       title,
       cmap="viridis",
       cmap_range=(0.10, 0.90),
   ):
       keys = sorted(bins.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bins[key], dtype=float)
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")
       ax.set_ylabel(r"Normalized $n_i(z)$")


   def square_triangles(x, y, size=0.42):
       x0, x1 = x - size, x + size
       y0, y1 = y - size, y + size

       tri1 = [(x0, y0), (x1, y0), (x0, y1)]
       tri2 = [(x1, y1), (x1, y0), (x0, y1)]
       border = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
       diag = [(x1, y0), (x0, y1)]

       return tri1, tri2, border, diag


   def draw_pair_cell(ax, row, col, color_i, color_j, size=0.42):
       tri1, tri2, border, diag = square_triangles(col, row, size=size)

       ax.add_patch(
           Polygon(
               tri1,
               closed=True,
               facecolor=color_i,
               edgecolor="none",
               alpha=0.65,
               zorder=3,
           )
       )

       ax.add_patch(
           Polygon(
               tri2,
               closed=True,
               facecolor=color_j,
               edgecolor="none",
               alpha=0.65,
               zorder=3,
           )
       )

       ax.add_patch(
           Polygon(
               border,
               closed=True,
               facecolor="none",
               edgecolor="k",
               linewidth=1.8,
               zorder=4,
           )
       )

       ax.plot(
           [diag[0][0], diag[1][0]],
           [diag[0][1], diag[1][1]],
           color="k",
           linewidth=1.2,
           zorder=5,
       )


   def draw_exclusion_overlay(ax, row, col, size=0.42):
       x0 = col - size
       y0 = row - size
       width = 2.0 * size
       height = 2.0 * size

       ax.add_patch(
           Rectangle(
               (x0, y0),
               width,
               height,
               facecolor="0.85",
               edgecolor="k",
               linewidth=1.5,
               alpha=0.65,
               zorder=10,
           )
       )
       ax.plot(
           [x0, x0 + width],
           [y0, y0 + height],
           color="k",
           linewidth=2.0,
           zorder=11,
       )
       ax.plot(
           [x0, x0 + width],
           [y0 + height, y0],
           color="k",
           linewidth=2.0,
           zorder=11,
       )


   def setup_square_pair_axes(ax, n_bins, title):
       ax.set_title(title)
       ax.set_xlim(-0.5, n_bins - 0.5)
       ax.set_ylim(n_bins - 0.5, -0.5)

       ax.set_xticks(range(n_bins))
       ax.set_yticks(range(n_bins))
       ax.set_xticklabels([f"{j + 1}" for j in range(n_bins)])
       ax.set_yticklabels([f"{i + 1}" for i in range(n_bins)])

       ax.set_xlabel("Lens bin $j$")
       ax.set_ylabel("Lens bin $i$")

       for k in range(n_bins + 1):
           ax.axhline(k - 0.5, color="k", linewidth=1.0, zorder=1)
           ax.axvline(k - 0.5, color="k", linewidth=1.0, zorder=1)

       for side in ["left", "right", "top", "bottom"]:
           ax.spines[side].set_visible(True)
           ax.spines[side].set_linewidth(1.8)

       ax.tick_params(
           axis="both",
           which="both",
           direction="in",
           top=True,
           right=True,
           width=1.5,
           length=5,
       )

       ax.grid(False)


   z = np.linspace(0.0, 2.4, 700)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.25,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   lens_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 5,
           "range": (0.2, 1.2),
       },
       "uncertainties": {
           "scatter_scale": [0.03, 0.045, 0.06, 0.08, 0.10],
           "mean_offset": [0.000, 0.005, 0.015, 0.025, 0.040],
           "outlier_frac": [0.02, 0.05, 0.10, 0.16, 0.22],
           "outlier_scatter_scale": [0.05, 0.07, 0.1, 0.12, 0.24],
           "outlier_mean_offset": [0.02, 0.05, 0.08, 0.1, 0.12],
       },
       "normalize_bins": True,
   }

   lens = NZTomography()
   lens_result = lens.build_bins(
       z=z,
       nz=nz,
       tomo_spec=lens_spec,
       include_tomo_metadata=True,
   )

   lens_keys = sorted(lens_result.bins.keys())
   n_lens = len(lens_keys)

   population = lens.population_stats(decimal_places=6)
   bin_fractions = population["fractions"]

   min_fraction_matrix = np.array(
       [
           [min(bin_fractions[i_key], bin_fractions[j_key]) for j_key in lens_keys]
           for i_key in lens_keys
       ],
       dtype=float,
   )

   threshold = 0.15

   candidate_pairs = [
       (i_key, j_key)
       for a, i_key in enumerate(lens_keys)
       for j_key in lens_keys[a:]
   ]

   retained_pairs = []
   excluded_pairs = []

   for i_key, j_key in candidate_pairs:
       i = lens_keys.index(i_key)
       j = lens_keys.index(j_key)

       if min_fraction_matrix[i, j] >= threshold:
           retained_pairs.append((i_key, j_key))
       else:
           excluded_pairs.append((i_key, j_key))

   lens_pos = {key: idx for idx, key in enumerate(lens_keys)}

   lens_colors = cmr.take_cmap_colors(
       "viridis",
       n_lens,
       cmap_range=(0, 1),
       return_fmt="hex",
   )

   fig = plt.figure(figsize=(15.2, 8.8))
   gs = fig.add_gridspec(
       2, 3, height_ratios=[1.0, 1.0],
       hspace=0.38, wspace=0.28)

   ax_top = fig.add_subplot(gs[0, :])
   ax0 = fig.add_subplot(gs[1, 0])
   ax1 = fig.add_subplot(gs[1, 1])
   ax2 = fig.add_subplot(gs[1, 2])

   # Top panel: lens bins in redshift space
   plot_one_sample(
       ax_top,
       z,
       lens_result.bins,
       "Lens tomographic bins",
   )

   fraction_lines = "\n".join(
       [rf"Bin {key + 1}: {100.0 * bin_fractions[key]:.1f}%" for key in lens_keys]
   )
   ax_top.text(
       0.98,
       0.96,
       fraction_lines,
       transform=ax_top.transAxes,
       ha="right",
       va="top",
       bbox=dict(
           boxstyle="round",
           facecolor="white",
           alpha=0.9,
           edgecolor="none",
       ),
   )

   # Panel 1: candidate symmetric topology
   setup_square_pair_axes(ax0, n_lens, "Candidate pairs")
   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_pair_cell(ax0, i, j, lens_colors[i], lens_colors[j])

   for i in range(n_lens):
       for j in range(i):
           ax0.add_patch(
               Rectangle(
                   (j - 0.5, i - 0.5),
                   1.0,
                   1.0,
                   facecolor="white",
                   edgecolor="none",
                   zorder=20,
               )
           )

   for k in range(n_lens + 1):
       ax0.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
       ax0.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

   # Panel 2: pairwise minimum-fraction diagnostic
   masked_min_fraction_matrix = min_fraction_matrix.copy()
   masked_min_fraction_matrix[np.tril_indices(n_lens, k=-1)] = np.nan

   cmap = plt.cm.viridis.copy()
   cmap.set_bad(color="white", alpha=0.0)

   ax1.imshow(
       masked_min_fraction_matrix,
       origin="upper",
       aspect="auto",
       cmap=cmap,
       alpha=0.65,
       interpolation="none",
       vmin=threshold,
       vmax=min_fraction_matrix.max(),
   )

   ax1.set_title("Pairwise min fraction")
   ax1.set_xticks(np.arange(n_lens))
   ax1.set_yticks(np.arange(n_lens))
   ax1.set_xticklabels([f"{k + 1}" for k in lens_keys])
   ax1.set_yticklabels([f"{k + 1}" for k in lens_keys])
   ax1.set_xlabel("Lens bin $j$")
   ax1.set_ylabel("Lens bin $i$")

   ax1.set_xticks(np.arange(-0.5, n_lens, 1), minor=True)
   ax1.set_yticks(np.arange(-0.5, n_lens, 1), minor=True)
   ax1.grid(which="minor", color="k", linestyle="-", linewidth=1.2)
   ax1.tick_params(which="minor", bottom=False, left=False)

   for side in ["left", "right", "top", "bottom"]:
       ax1.spines[side].set_visible(True)
       ax1.spines[side].set_linewidth(1.8)

   ax1.tick_params(
       axis="both",
       which="both",
       direction="in",
       top=True,
       right=True,
       width=1.5,
       length=5,
   )

   for i in range(n_lens):
       for j in range(i, n_lens):
           value = min_fraction_matrix[i, j]
           txt = f"{100.0 * value:.1f}"
           text_color = "k" if value >= threshold else "white"
           ax1.text(
               j,
               i,
               txt,
               ha="center",
               va="center",
               color=text_color,
               zorder=5,
           )

   # Panel 3: retained pairs after population cut
   setup_square_pair_axes(
       ax2,
       n_lens,
       rf"Retained pairs $\geq {100 * threshold:.0f}\%$",
   )

   for i_key, j_key in candidate_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_pair_cell(ax2, i, j, lens_colors[i], lens_colors[j])

   for i in range(n_lens):
       for j in range(i):
           ax2.add_patch(
               Rectangle(
                   (j - 0.5, i - 0.5),
                   1.0,
                   1.0,
                   facecolor="white",
                   edgecolor="none",
                   zorder=20,
               )
           )

   for i_key, j_key in excluded_pairs:
       i = lens_pos[i_key]
       j = lens_pos[j_key]
       draw_exclusion_overlay(ax2, i, j)

   for k in range(n_lens + 1):
       ax2.axhline(k - 0.5, color="k", linewidth=1.0, zorder=21)
       ax2.axvline(k - 0.5, color="k", linewidth=1.0, zorder=21)

   plt.tight_layout()
