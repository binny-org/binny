.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin diagnostics
======================

Tomographic binning produces a set of redshift-distribution curves
:math:`n_i(z)` describing the galaxies assigned to each bin. While
per-bin summaries describe the internal properties of each curve,
many practical questions instead involve **relationships between bins**.

Examples include:

- how strongly two bins overlap in redshift,
- how much of one bin spills into the nominal range of another,
- how similar two bin curves are in overall shape,
- how bins from different tomographic samples relate to each other.

Binny refers to these quantities collectively as **bin diagnostics**.
They are cross-bin statistics designed to quantify **coupling between
tomographic bins**.

This page introduces the main classes of diagnostics provided by
Binny and explains how they should be interpreted.

For practical examples of these diagnostics in action, including
visualizations and comparisons between different tomographic binning
schemes, see the :doc:`Bin diagnostics examples <../examples/bin_diagnostics>`.


Why cross-bin diagnostics are useful
------------------------------------

Tomographic bins are often designed to isolate galaxies into relatively
separate redshift slices. In practice, however, redshift uncertainty,
photo-z scatter, and outliers blur the boundaries between bins.

As a result:

- bins that are nominally separated may still share redshift support,
- galaxies assigned to one bin may fall within the intended interval of
  another,
- two bins may have similar shapes even if they are centered at
  different redshifts.

Cross-bin diagnostics help quantify these effects.

In general, these diagnostics answer questions such as:

- **Overlap:** how much redshift support two bins share.
- **Leakage:** how much of one bin falls into another bin's nominal
  redshift interval.
- **Correlation:** how similar two bin curves are when viewed as
  functions of redshift.

These quantities help determine whether a tomographic binning scheme
produces well-separated bins or instead exhibits substantial coupling.


Similarity and distance measures
--------------------------------

Many cross-bin diagnostics compare two bin curves
:math:`n_i(z)` and :math:`n_j(z)`.

Some metrics quantify **similarity**, meaning larger values correspond
to more similar bins. Others quantify **distance**, meaning larger
values correspond to more distinct bins.

For clarity, it is helpful to keep these two categories separate.

Similarity measures
~~~~~~~~~~~~~~~~~~~

Similarity metrics increase when two bin curves resemble each other
more closely.

Two commonly used examples are:

**Min overlap**

The min-overlap metric measures the integral of the pointwise minimum
of the two curves:

.. math::

   O_{ij} = \int \min[n_i(z), n_j(z)]\,\mathrm{d}z.

If both curves are normalized, then

.. math::

   0 \le O_{ij} \le 1.

Interpretation:

- :math:`O_{ij} = 1` means the two curves are identical,
- :math:`O_{ij} \approx 0` means they have little shared support.

This is one of the most direct ways to measure how much two bins occupy
the same redshift region.

**Cosine similarity**

Cosine similarity compares two curves as vectors sampled on the same
redshift grid:

.. math::

   C_{ij}
   =
   \frac{
      \int n_i(z)\,n_j(z)\,\mathrm{d}z
   }{
      \sqrt{
         \int n_i(z)^2\,\mathrm{d}z
         \int n_j(z)^2\,\mathrm{d}z
      }
   }.

For nonnegative curves,

.. math::

   0 \le C_{ij} \le 1.

Cosine similarity measures how closely aligned the shapes of two curves
are, even if their amplitudes differ.


Distance measures
~~~~~~~~~~~~~~~~~

Distance metrics instead measure how different two distributions are.
In these cases **smaller values correspond to greater similarity**.

Several probability-distance measures are commonly used.

**Jensen–Shannon distance**

The Jensen–Shannon distance compares two probability distributions
derived from the bin curves. It is based on the symmetrized
Kullback–Leibler divergence.

If :math:`p` and :math:`q` are probability vectors representing segment
masses of two bins, the Jensen–Shannon distance is

.. math::

   D_{\mathrm{JS}}(p,q)
   =
   \sqrt{
      \frac{1}{2} KL(p||m)
      +
      \frac{1}{2} KL(q||m)
   },

where :math:`m = (p+q)/2`.

Properties:

- :math:`0` indicates identical distributions,
- larger values indicate greater dissimilarity.

**Hellinger distance**

The Hellinger distance measures geometric separation between two
probability distributions:

.. math::

   H(p,q)
   =
   \sqrt{
      \frac{1}{2}
      \sum_k
      \left(\sqrt{p_k}-\sqrt{q_k}\right)^2
   }.

Values lie between 0 and 1.

**Total variation distance**

Total variation distance measures the maximum discrepancy between two
probability distributions:

.. math::

   D_{\mathrm{TV}}(p,q)
   =
   \frac{1}{2}\sum_k |p_k - q_k|.

It also lies between 0 and 1.

These distance metrics compare distributions based on **probability
mass vectors**, which represent how bin weight is distributed across
segments of the redshift grid.


Leakage diagnostics
-------------------

Overlap metrics measure shared support between bin curves, but they do
not directly account for **nominal bin edges**.

Leakage diagnostics instead compare bin curves to the **intended
redshift intervals** of other bins.

Suppose bin :math:`j` is defined by nominal edges
:math:`[z_j^{\min}, z_j^{\max}]`. The leakage from bin :math:`i` into
bin :math:`j` is

.. math::

   L_{ij}
   =
   \frac{
      \int_{z_j^{\min}}^{z_j^{\max}} n_i(z)\,\mathrm{d}z
   }{
      \int n_i(z)\,\mathrm{d}z
   }.

Interpretation:

- :math:`L_{ii}` measures **completeness** — the fraction of bin
  :math:`i` that remains inside its own interval.
- :math:`1 - L_{ii}` measures **contamination** — the fraction that
  falls outside the intended interval.
- off-diagonal entries :math:`L_{ij}` quantify how much of bin
  :math:`i` lies inside bin :math:`j`’s interval.

Leakage matrices are therefore **directional**: the amount of bin
:math:`i` leaking into bin :math:`j` need not equal the reverse.


Pearson correlation
-------------------

Pearson correlation measures similarity between two curves in terms of
their fluctuations across the redshift grid.

For curves :math:`f(z)` and :math:`g(z)`, the Pearson correlation is

.. math::

   \rho(f,g)
   =
   \frac{
      \mathrm{cov}(f,g)
   }{
      \sigma_f \sigma_g
   }.

Here the covariance and variances are computed using trapezoid
integration weights.

Properties:

- :math:`\rho = 1` indicates perfectly correlated curves,
- :math:`\rho = 0` indicates no correlation,
- :math:`\rho = -1` indicates perfect anticorrelation.

Unlike overlap or leakage, Pearson correlation focuses on **shape
similarity** rather than shared support in redshift.


Between-sample diagnostics
--------------------------

In many analyses, two tomographic samples are used simultaneously. A
common example is galaxy–galaxy lensing, where one defines **lens bins**
and **source bins**.

In these cases, diagnostics can compare bins **between two samples**
rather than within one sample.

The same families of metrics apply:

- overlap between bins from the two samples,
- Pearson correlations between bin curves,
- interval-based mass transfer between bins.

Between-sample matrices are generally **rectangular** rather than
square, because the two samples can contain different numbers of bins.

These diagnostics help reveal which bin pairs across the two samples
are most strongly coupled in redshift.


Interpreting diagnostic matrices
--------------------------------

Cross-bin diagnostics are typically displayed as matrices.

For within-sample diagnostics:

- rows and columns correspond to the same set of bins,
- the matrix is symmetric for similarity metrics,
- the diagonal entries represent comparisons of bins with themselves.

For between-sample diagnostics:

- rows correspond to bins from one sample,
- columns correspond to bins from another sample,
- the matrix is generally rectangular.

Some general patterns are useful to look for:

- **strong diagonals** usually indicate well-separated bins,
- **large off-diagonal overlap** indicates shared redshift support,
- **large off-diagonal leakage** indicates spillover into neighboring
  bins,
- **strong off-diagonal correlations** indicate similar bin shapes.


Relationship to the examples
----------------------------

This page introduces the concepts behind Binny's diagnostic metrics.

For worked examples showing how these quantities behave in practice,
including visualizations of overlap matrices, leakage patterns, and
similarity comparisons between tomographic bins, see the
:doc:`Bin diagnostics examples <../examples/bin_diagnostics>`.

A useful way to think about the split is:

- this page explains **what the diagnostic quantities mean**,
- the examples page shows **how they behave for real binning choices**.

Together they provide both the theoretical interpretation and the
practical diagnostics needed to inspect tomographic bin coupling.


Notes
-----

- Similarity metrics increase as bins become more alike, whereas
  distance metrics decrease.
- Leakage diagnostics depend on the **nominal bin edges**, whereas
  overlap metrics depend only on the bin curves.
- Pearson correlation captures **shape similarity**, not necessarily
  direct redshift overlap.
- Between-sample diagnostics extend the same ideas to comparisons
  between different tomographic samples.