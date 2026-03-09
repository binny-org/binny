.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin-index combinations
=============================

This page shows practical examples of selecting bin-index pairs and tuples with
:class:`binny.NZTomography` and :class:`binny.BinComboFilter`.

The examples below illustrate three common ideas:

- build a candidate set of tuples from a named topology,
- filter those tuples using score-based or curve-based criteria,
- return the surviving index tuples for later use in an analysis.

For the theory behind these selections, see :doc:`../theory/bin_combos`.


Basic workflow
--------------

At the API level, the most common entry point is
:meth:`binny.NZTomography.bin_combo_filter`.

A selection is written as a mapping with two optional parts:

- a ``topology`` block, which builds the initial tuple set,
- a ``filters`` list, which applies one or more ordered selection steps.

A minimal pattern looks like this:

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
       "filters": [
           {"name": "score_relation", "score": "peak", "relation": "lt"},
       ],
   }

   selected = tomo.bin_combo_filter(spec)

This returns a list of tuples such as ``[(0, 1), (0, 2), (1, 2)]``.


Example setup
-------------

The examples below assume a tomography object has already been built.

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

The examples then operate on the cached bin curves in ``tomo``.


Example 1: all ordered pairs
----------------------------

To generate all ordered bin pairs from a single sample:

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
   }

   pairs = tomo.bin_combo_filter(spec)

This includes all pairs ``(i, j)`` built from the available bin keys.

For four bins, that would typically return

.. code-block:: python

   [
       (0, 0), (0, 1), (0, 2), (0, 3),
       (1, 0), (1, 1), (1, 2), (1, 3),
       (2, 0), (2, 1), (2, 2), (2, 3),
       (3, 0), (3, 1), (3, 2), (3, 3),
   ]


Example 2: upper-triangular pairs
---------------------------------

A common choice is to keep only pairs with :math:`i \le j`.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
   }

   pairs = tomo.bin_combo_filter(spec)

This avoids duplicate symmetric pairs when the ordering is not important.


Example 3: off-diagonal pairs only
----------------------------------

To exclude auto-pairs and keep only cross-pairs:

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_off_diagonal"},
   }

   pairs = tomo.bin_combo_filter(spec)

This is useful when only cross-bin combinations are needed.


Example 4: require peak ordering
--------------------------------

You can filter pairs by comparing score summaries of the selected curves.

For example, to keep only pairs whose second bin peaks at higher redshift than
the first:

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

   pairs = tomo.bin_combo_filter(spec)

Here:

- ``score="peak"`` uses the peak location of each curve,
- ``pos_a`` and ``pos_b`` refer to tuple slots,
- ``relation="gt"`` requires slot 1 to have a larger score than slot 0.

This is a simple way to enforce redshift ordering.


Example 5: minimum peak separation
----------------------------------

To keep only pairs whose peak locations differ by at least some amount:

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

   pairs = tomo.bin_combo_filter(spec)

This is useful when excluding pairs whose effective redshift support is too
similar.


Example 6: signed score differences
-----------------------------------

If the sign of the difference matters, use ``score_difference`` instead of
absolute separation.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "score_difference",
               "score": "mean",
               "pos_a": 0,
               "pos_b": 1,
               "min_diff": 0.1,
               "max_diff": 0.5,
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

This keeps pairs satisfying

.. math::

   0.1 \le s(i_1) - s(i_0) \le 0.5,

using the mean-location score.


Example 7: consistent ordering under two summaries
--------------------------------------------------

To require that two different score definitions agree on the ordering:

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "score_consistency",
               "score1": "peak",
               "score2": "mean",
               "relation": "gt",
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

This can be useful when you want a selection that is less sensitive to the
choice of summary statistic.


Example 8: width compatibility
------------------------------

To require that two bins have similar credible widths:

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

   pairs = tomo.bin_combo_filter(spec)

This uses the built-in credible-width score and keeps only pairs whose widths
are compatible up to the requested ratio.


Example 9: low-overlap cross-pairs
----------------------------------

To exclude pairs with large curve overlap:

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

This keeps only pairs with overlap fraction at most ``0.25``.

This is often a natural way to remove strongly redundant pairs.


Example 10: overlap coefficient
-------------------------------

A closely related option is the overlap coefficient:

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

This can be useful when you want a normalized measure of how much one curve is
contained inside another.


Example 11: exclude nearly empty curves
---------------------------------------

If your curves are not individually normalized, you may want to remove tuples
containing very small integrated weight.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_all"},
       "filters": [
           {
               "name": "curve_norm_threshold",
               "threshold": 0.01,
               "compare": "ge",
               "mode": "all",
           }
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

With ``mode="all"``, every slot in the tuple must pass the threshold.
With ``mode="any"``, only one slot must pass.


Example 12: chaining multiple filters
-------------------------------------

Selections become most useful when several filters are applied in sequence.

.. code-block:: python

   spec = {
       "topology": {"name": "pairs_upper_triangle"},
       "filters": [
           {
               "name": "score_relation",
               "score": "peak",
               "relation": "gt",
           },
           {
               "name": "score_separation",
               "score": "mean",
               "min_sep": 0.15,
               "absolute": True,
           },
           {
               "name": "overlap_fraction",
               "threshold": 0.3,
               "compare": "le",
           },
       ],
   }

   pairs = tomo.bin_combo_filter(spec)

This first builds upper-triangular pairs, then keeps only those that satisfy
all three conditions in order.


Example 13: between-sample pairs
--------------------------------

You can also compare bins from two different tomography objects.

.. code-block:: python

   lens = NZTomography()
   lens.build_bins(
       z=z,
       nz=nz,
       tomo_spec={
           "kind": "photoz",
           "bins": {"scheme": "equidistant", "n_bins": 3},
       },
   )

   source = NZTomography()
   source.build_bins(
       z=z,
       nz=nz,
       tomo_spec={
           "kind": "photoz",
           "bins": {"scheme": "equipopulated", "n_bins": 4},
       },
   )

   spec = {
       "topology": {"name": "pairs_cartesian"},
       "filters": [
           {
               "name": "score_relation",
               "score": "mean",
               "relation": "gt",
           }
       ],
   }

   selected = lens.bin_combo_filter(spec, other=source)

In this case:

- slot 0 uses the bins from ``lens``,
- slot 1 uses the bins from ``source``,
- ``pairs_cartesian`` builds all cross-sample pairs.

This is useful for selections involving two distinct bin families.


Example 14: higher-order tuples
-------------------------------

The same interface also supports triplets and higher-order tuples.

For example, to build all nondecreasing triplets from one sample:

.. code-block:: python

   from binny.correlations.bin_combo_filter import BinComboFilter

   f = BinComboFilter(z=tomo.z, curves=[tomo.bins, tomo.bins, tomo.bins])

   triplets = (
       f.set_topology("tuples_nondecreasing")
        .values()
   )

Or with a selection spec:

.. code-block:: python

   from binny.correlations.bin_combo_filter import BinComboFilter

   f = BinComboFilter(z=tomo.z, curves=[tomo.bins, tomo.bins, tomo.bins])

   spec = {
       "topology": {"name": "tuples_nondecreasing"},
       "filters": [
           {
               "name": "curve_norm_threshold",
               "threshold": 0.01,
               "compare": "ge",
               "mode": "all",
           }
       ],
   }

   triplets = f.select(spec).values()

This is the more general interface when you want more than two tuple slots.


Example 15: custom metric kernels
---------------------------------

You can register your own metric kernel and reference it from a selection spec.

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

This is useful when your notion of similarity or separation is not captured by
the built-in overlap summaries.


Selection-spec reference
------------------------

Supported topology names include:

- ``pairs_all``
- ``pairs_upper_triangle``
- ``pairs_lower_triangle``
- ``pairs_diagonal``
- ``pairs_off_diagonal``
- ``pairs_cartesian``
- ``tuples_all``
- ``tuples_nondecreasing``
- ``tuples_diagonal``
- ``tuples_cartesian``

Supported built-in filter names include:

- ``overlap_fraction``
- ``overlap_coefficient``
- ``metric``
- ``score_relation``
- ``score_separation``
- ``score_difference``
- ``score_consistency``
- ``width_ratio``
- ``curve_norm_threshold``


Notes
-----

A few practical points are worth keeping in mind:

- ``pos_a`` and ``pos_b`` refer to tuple slots, not necessarily to specific
  physical sample types.
- filters are applied in order, so later filters act on the already reduced
  tuple list.
- for between-sample filtering, both tomography objects must share the same
  redshift grid.
- the high-level :meth:`binny.NZTomography.bin_combo_filter` interface is the
  easiest option for pair selections, while direct use of
  :class:`binny.BinComboFilter` is more flexible for higher-order tuples.