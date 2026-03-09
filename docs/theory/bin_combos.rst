.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin-index combinations
=============================

In many tomographic analyses, not every possible combination of bins is used.
Instead, a set of candidate bin-index tuples is first constructed and then
filtered to retain only those that satisfy some structural or statistical
criterion.

Examples include:

- keeping only auto-correlations or only cross-correlations,
- selecting pairs whose peaks are sufficiently separated,
- excluding combinations with strong overlap,
- requiring a consistent ordering between bins across different summary
  statistics,
- retaining only tuples built from sufficiently populated curves.

Binny provides this functionality through
:class:`binny.BinComboFilter`, which combines two ideas:

- **topology builders**, which generate collections of index tuples,
- **filters**, which remove tuples according to score-based or
  curve-based criteria.

This page describes the mathematical and conceptual structure behind that
workflow.


Motivation
----------

Suppose a tomographic construction produces a family of bin curves
:math:`n_i(z)` indexed by :math:`i`. In practice, it is often useful to work
not only with individual bins, but with ordered tuples of bin indices such as

.. math::

   (i, j), \qquad (i, j, k), \qquad \text{or more generally} \qquad
   (i_1, i_2, \dots, i_r).

These tuples can represent, for example, pairs of bins to be correlated,
triplets to be compared, or more general multi-slot index combinations.

The main task is therefore not only to build tuples, but also to decide which
tuples are meaningful for a given analysis. That decision may depend on:

- the tuple topology itself,
- scalar summaries of the selected curves,
- direct comparisons between the curves,
- or combinations of these conditions.

:class:`binny.BinComboFilter` is designed to make these selections explicit,
reproducible, and easy to configure.


Tuple topologies
----------------

A **topology** is a rule for generating tuples from one or more index sets.

For bin pairs built from a single index set :math:`K`, common examples are:

.. math::

   \{(i,j) : i,j \in K\},

for all ordered pairs, or restricted variants such as

.. math::

   \{(i,j) : i \le j\}, \qquad
   \{(i,j) : i < j\}, \qquad
   \{(i,i)\}.

These correspond, respectively, to all pairs, upper-triangular pairs,
off-diagonal pairs, and diagonal pairs.

When two tuple positions draw from different index sets
:math:`K_0` and :math:`K_1`, a Cartesian product can be formed instead,

.. math::

   \{(i,j) : i \in K_0,\; j \in K_1\}.

More generally, for :math:`r` tuple positions, tuples of length :math:`r`
can be constructed using either a shared key set or one key set per slot.

In Binny, these constructions are handled by named topology builders such as

- ``pairs_all``,
- ``pairs_upper_triangle``,
- ``pairs_lower_triangle``,
- ``pairs_diagonal``,
- ``pairs_off_diagonal``,
- ``pairs_cartesian``,
- ``tuples_all``,
- ``tuples_nondecreasing``,
- ``tuples_diagonal``,
- ``tuples_cartesian``.

Conceptually, topology answers the question:

**Which tuples are even allowed before any filtering is applied?**


Tuple slots and stored curves
-----------------------------

A tuple of length :math:`r` has :math:`r` positions, or **slots**.
For a tuple

.. math::

   (i_0, i_1, \dots, i_{r-1}),

slot :math:`p` stores the index :math:`i_p`.

In :class:`binny.BinComboFilter`, each slot is associated with a mapping from
index to curve. If the tuple has :math:`r` slots, the filter stores
:math:`r` such mappings,

.. math::

   \mathcal{C}_0,\; \mathcal{C}_1,\; \dots,\; \mathcal{C}_{r-1},

where :math:`\mathcal{C}_p(i)` denotes the curve associated with index
:math:`i` in slot :math:`p`.

All curves are evaluated on a shared coordinate grid :math:`z`.

This means that a tuple does not merely specify indices. It also selects
one concrete curve per slot, which can then be summarized or compared.


Score-based summaries
---------------------

Many filters operate not directly on the full curves, but on scalar
**scores** derived from each curve.

If :math:`c(z)` is a curve, a score is a map

.. math::

   s[c] \in \mathbb{R}.

Binny includes several score definitions:

- **peak location**, the coordinate where the curve reaches its maximum,
- **mean location**, the first moment of the normalized curve,
- **median location**, the coordinate dividing the integrated mass in half,
- **credible width**, the width of a chosen credible-mass interval.

For a family of curves :math:`c_i(z)`, these define scalar maps

.. math::

   i \mapsto s_i.

Once scores have been computed for each slot, tuples can be filtered by
relations between slot values.

For example, a pair :math:`(i,j)` may be kept if

.. math::

   s_j > s_i,

or if the absolute separation satisfies

.. math::

   |s_j - s_i| \ge \Delta_{\min}.

In plain terms, score-based filters reduce each curve to a simple number and
then compare those numbers across tuple positions.


Score relations, differences, and consistency
---------------------------------------------

Several useful tuple selections can be expressed in terms of score
comparisons.

A **score relation** imposes an ordering such as

.. math::

   s(i_{p_b}) < s(i_{p_a}),
   \qquad
   s(i_{p_b}) \le s(i_{p_a}),
   \qquad
   s(i_{p_b}) > s(i_{p_a}),
   \qquad
   s(i_{p_b}) \ge s(i_{p_a}).

A **score difference** keeps tuples whose signed difference lies within a
window,

.. math::

   d = s(i_{p_b}) - s(i_{p_a}),
   \qquad
   d_{\min} \le d \le d_{\max}.

A **score separation** uses the same idea but often emphasizes magnitude,
for example through

.. math::

   |s(i_{p_b}) - s(i_{p_a})|.

A **score consistency** condition requires that the same ordering hold under
two different score definitions. For example, one may require both the peak
and the mean location of slot :math:`p_b` to lie above those of slot
:math:`p_a`.

This is useful when the goal is to define selections that are not sensitive
to the choice of summary statistic.


Width compatibility
-------------------

A special case of score-based filtering compares the widths of curves across
different tuple positions.

If :math:`w_i` denotes a width summary for index :math:`i`, one can require
that two slots satisfy a ratio bound such as

.. math::

   \frac{w(i_{p_a})}{w(i_{p_b})} \le R_{\max},

or, in symmetric form, that neither width exceeds the other by more than a
factor :math:`R_{\max}`.

This is useful when comparing bins that should have broadly similar
effective spread, even if their central locations differ.


Curve-based metrics
-------------------

Other filters operate on the selected curves directly.

A curve-based metric is a function

.. math::

   M(c_1, c_2, \dots, c_r) \in \mathbb{R},

evaluated on the curves chosen by a tuple.

For a tuple :math:`(i_0, \dots, i_{r-1})`, this becomes

.. math::

   M\bigl(\mathcal{C}_0(i_0), \mathcal{C}_1(i_1), \dots,
   \mathcal{C}_{r-1}(i_{r-1})\bigr).

The tuple is then kept if the metric satisfies a threshold relation such as

.. math::

   M \ge t
   \qquad \text{or} \qquad
   M \le t.

This is the most general filtering mode, since it allows arbitrary
curve-level comparisons.


Overlap-based metrics
---------------------

Two built-in curve-based metrics are especially useful for tomography.

The first is a **minimum-overlap fraction**, which measures how much shared
area exists between selected curves relative to a chosen normalization.

The second is the **overlap coefficient**, which quantifies the degree to
which one curve lies inside another in terms of common integrated support.

These metrics are useful when deciding whether two bins are well separated
or substantially mixed.

In plain terms:

- a **small overlap metric** indicates that the selected curves are well
  separated,
- a **large overlap metric** indicates that the curves trace similar
  redshift support and may be difficult to distinguish cleanly.

This makes overlap-based filters particularly natural for selecting
cross-bin combinations or excluding strongly redundant tuples.


Curve norm thresholds
---------------------

Some selections should exclude tuples containing curves with very small total
weight. For a curve :math:`c(z)`, define its norm as

.. math::

   N[c] = \int c(z)\,dz.

A tuple can then be filtered by requiring that these norms satisfy a
threshold condition, either for all slots or for at least one slot.

This is useful because a tuple involving nearly empty curves may be
mathematically valid but practically uninformative.

If the curves are individually normalized, these norms may all be similar.
If they are not, norm-based filtering can act as a simple population-aware
screen.


Custom metrics
--------------

Binny also supports user-defined metric kernels.
Conceptually, this makes it possible to define a custom scalar function of
the selected curves and then apply threshold-based filtering to that derived
quantity.

This is useful when a problem requires a specialized notion of similarity,
distance, asymmetry, or alignment that is not captured by the built-in
overlap summaries.


Selection as a two-stage procedure
----------------------------------

The overall filtering workflow can be viewed as a two-stage map:

1. build an initial tuple set using a topology,
2. apply one or more filters to retain only the desired tuples.

Schematically,

.. math::

   \mathcal{T}_{\mathrm{raw}}
   \xrightarrow{\text{filters}}
   \mathcal{T}_{\mathrm{selected}}.

The order of filters matters in practice because each filter acts on the
current working tuple list. This makes the filtering process naturally
compositional.

In Binny, this sequence can be expressed either through explicit method
calls or through a YAML-friendly selection specification.


YAML-friendly selection specifications
--------------------------------------

For reproducible workflows, Binny allows tuple selections to be written as a
structured specification containing

- an optional topology block,
- an ordered list of filter blocks.

This allows tuple construction and filtering logic to be moved out of
ad hoc analysis code and into configuration.

Conceptually, this is useful because it makes tuple selection part of the
documented analysis setup rather than an implicit coding choice.


Use in cosmological analyses
----------------------------

In cosmological tomography, the tomographic bin curves themselves are only one
part of the construction. A further step is deciding which bin-index
combinations will actually be carried forward into forecasts or measured data
vectors.

For example, a tomographic observable may involve pairs such as
:math:`(i,j)` between source bins, lens bins, or between distinct lens and
source samples. In some cases all formally allowed combinations are used, but
in many practical workflows only a subset is retained.

This selection can reflect the structure of the observable itself, such as
keeping only symmetric pairs or only cross-sample pairs, but it can also be
guided by statistical properties of the bins. For instance, it may be useful
to exclude combinations with very strong overlap, require a minimum separation
in peak or mean redshift, or retain only tuples whose widths are sufficiently
compatible.

In that sense, bin-combination filtering sits between tomographic
construction and downstream inference. The bins define the available building
blocks, while the tuple filter defines which combinations of those blocks are
considered useful, stable, or relevant for forecasting and later analysis.


Interpretation
--------------

Bin-index combination filtering is not a separate tomographic model.
Rather, it is a way of selecting which bin combinations are considered
relevant once curves already exist.

Its role is therefore organizational and diagnostic:

- it makes tuple construction explicit,
- it provides transparent selection criteria,
- it helps remove redundant or poorly separated combinations,
- and it supports reproducible downstream analyses.

In practice, this is most useful whenever a workflow needs more structure
than "use all pairs" but less bespoke code than writing manual tuple filters
for every case.