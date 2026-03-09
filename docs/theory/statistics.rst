.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Bin summaries
====================

Tomographic binning does not only produce a collection of bin curves
:math:`n_i(z)`. It also produces a set of useful summary quantities that
help characterize how those bins are distributed in redshift and how
they partition the galaxy sample.

In Binny, these summaries are separated into two broad families:

- **shape statistics**, which depend only on the shape of each bin curve,
- **population statistics**, which depend on how galaxies are distributed
  across bins.

This distinction is important because many tomographic workflows use bin
curves that are individually normalized. Once each bin is normalized on
its own, the relative number of galaxies in different bins is no longer
encoded in the curve amplitudes. Shape information remains valid, but
population information must instead come from tomography metadata.

This page introduces the main summary quantities provided by Binny and
explains how they should be interpreted.

For practical examples of these statistics in action, including
visualizations and comparisons between different binning schemes,
see the :doc:`Bin summaries examples <../examples/bin_summaries>`.


Why separate shape and population statistics?
---------------------------------------------

Suppose a tomographic construction returns bin curves
:math:`n_i(z)` for bins :math:`i=1,\dots,N`.

Some summaries depend only on the internal structure of each curve. For
example, one may ask:

- where the bin is centered in redshift,
- how broad it is,
- whether it is symmetric or skewed,
- whether it contains more than one peak.

These are **shape statistics**. They remain meaningful even if every bin
curve is rescaled or normalized independently.

Other summaries depend on the relative amount of galaxy population
assigned to each bin. For example, one may ask:

- what fraction of galaxies lies in each bin,
- what effective galaxy density belongs to each bin,
- what effective galaxy count belongs to each bin.

These are **population statistics**. They cannot, in general, be
recovered from individually normalized bin curves and should instead be
taken from the metadata returned by the tomographic builder.

In practice, this means:

- use **shape statistics** to describe what each bin looks like,
- use **population statistics** to describe how the total sample is
  divided among bins.


Shape statistics
----------------

Shape statistics summarize the redshift structure of each individual bin
curve. They are designed to answer questions such as:

- Where is the bin located?
- How broad is it?
- Is it symmetric?
- Does it show secondary structure?

Throughout this section, let :math:`n_i(z)` denote the curve of bin
:math:`i` defined on a redshift grid :math:`z`. The curve is treated as
a nonnegative weight function and does not need to be normalized in
advance.


Moments and width summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural first set of summaries is given by weighted moments of the bin
curve.

The weighted mean redshift of bin :math:`i` is

.. math::

   \bar z_i
   =
   \frac{\int z\,n_i(z)\,\mathrm{d}z}{\int n_i(z)\,\mathrm{d}z}.

This gives a representative center of the bin, but it can shift toward
extended tails if the distribution is asymmetric.

The weighted variance is

.. math::

   \sigma_{z,i}^2
   =
   \frac{\int (z-\bar z_i)^2 n_i(z)\,\mathrm{d}z}{\int n_i(z)\,\mathrm{d}z},

with corresponding standard deviation :math:`\sigma_{z,i}`. This is a
useful measure of spread, although it is not always the most robust
choice when the bin is strongly skewed or contains outliers.

Higher standardized moments describe shape asymmetry and tail structure:

.. math::

   \mathrm{skewness}_i
   =
   \frac{\int (z-\bar z_i)^3 n_i(z)\,\mathrm{d}z}
        {\sigma_{z,i}^3 \int n_i(z)\,\mathrm{d}z},

.. math::

   \mathrm{kurtosis}_i
   =
   \frac{\int (z-\bar z_i)^4 n_i(z)\,\mathrm{d}z}
        {\sigma_{z,i}^4 \int n_i(z)\,\mathrm{d}z}
   - 3.

A positive skewness indicates a longer high-redshift tail, while a
negative skewness indicates a longer low-redshift tail. The kurtosis
measures whether the distribution is more sharply peaked or more
heavy-tailed than a Gaussian reference.

Binny also provides robust width summaries based on quantiles rather
than moments. Two especially useful ones are:

- the **interquartile range**
  :math:`\mathrm{IQR} = q_{75} - q_{25}`,
- the **central 68\% width**
  :math:`w_{68} = q_{84} - q_{16}`.

These are often easier to interpret than the standard deviation when bin
curves are skewed or mildly non-Gaussian.


Quantiles and representative centers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A weighted quantile :math:`q_p` is defined through the cumulative
integral of the bin curve. It identifies the redshift below which a
fraction :math:`p` of the bin weight lies.

Common examples are:

- :math:`q_{50}`, the weighted median,
- :math:`q_{16}` and :math:`q_{84}`, which bracket the central
  68\% interval,
- :math:`q_{25}` and :math:`q_{75}`, which define the interquartile
  range.

Quantiles are especially useful because they remain meaningful even when
the bin is asymmetric or contains extended tails.

From these quantities one can define different notions of a
representative bin center:

- **mean**: sensitive to the full weight distribution,
- **median**: robust to tails and outliers,
- **mode**: the redshift at which the bin curve reaches its maximum,
- **percentile-based centers** such as :math:`p50`, :math:`p16`, or
  :math:`p84`.

These choices are not interchangeable. If the mean, median, and mode all
lie close together, the bin is usually fairly compact and symmetric. If
they differ significantly, the bin likely has skewness, leakage, or
secondary structure.

This distinction matters in cosmology because many nuisance or
astrophysical ingredients are defined per tomographic bin and evaluated
at a characteristic redshift. Examples include intrinsic-alignment
amplitudes, galaxy-bias parameters, and magnification-bias terms.


Peak diagnostics
~~~~~~~~~~~~~~~~

Some bin curves are close to unimodal, while others can contain visible
secondary structure. This can happen because of photometric-redshift
outliers, leakage from neighboring bins, or more complicated response
models.

Binny provides simple peak diagnostics based on local maxima of the bin
curve. The main quantities are:

- the **mode**, i.e. the location of the global maximum,
- the **mode height**, i.e. the value of the curve at that maximum,
- the **number of peaks** above a chosen relative threshold,
- the **second-peak ratio**, defined as the height of the
  second-highest peak divided by the height of the highest one.

The second-peak ratio is particularly useful as a compact indicator of
multimodality:

- a value near **0** indicates weak or absent secondary structure,
- a larger value indicates that a secondary feature is more prominent.

This should be interpreted as a structural diagnostic, not as a direct
statement about the physical origin of the feature.


Tail asymmetry
~~~~~~~~~~~~~~

A convenient way to summarize asymmetry without using higher moments is
to compare the upper and lower spreads around the median.

Using the 16th, 50th, and 84th percentiles, one can define

.. math::

   A_i
   =
   \frac{q_{84,i} - q_{50,i}}{q_{50,i} - q_{16,i}}.

This compares the upper tail width to the lower tail width.

Interpretation is straightforward:

- :math:`A_i = 1` indicates symmetric upper and lower spreads,
- :math:`A_i > 1` indicates a longer high-redshift tail,
- :math:`A_i < 1` indicates a longer low-redshift tail.

This statistic is often easier to interpret than skewness because it is
tied directly to percentile widths.


In-range fraction
~~~~~~~~~~~~~~~~~

If nominal bin edges are known, one can ask how much of a bin curve
actually remains inside its intended redshift interval.

For bin :math:`i` with nominal interval :math:`[z_i^{\min}, z_i^{\max}]`,
the in-range fraction is

.. math::

   f_i^{\mathrm{in}}
   =
   \frac{
      \int_{z_i^{\min}}^{z_i^{\max}} n_i(z)\,\mathrm{d}z
   }{
      \int n_i(z)\,\mathrm{d}z
   }.

This quantity measures how cleanly the bin remains localized with
respect to its intended redshift range.

Interpretation:

- :math:`f_i^{\mathrm{in}} \approx 1` means the bin is well confined to
  its nominal interval,
- smaller values indicate more spillover outside the intended range.

In photometric tomography this is a useful summary of how strongly
scatter, outliers, or leakage blur the connection between nominal bin
edges and the actual redshift content of the bin.


Nominal bin widths
~~~~~~~~~~~~~~~~~~

When bin edges are supplied, Binny can also summarize the widths of the
nominal redshift intervals themselves.

For edges :math:`e_0, e_1, \dots, e_N`, the nominal width of bin
:math:`i` is simply

.. math::

   \Delta z_i = e_{i+1} - e_i.

These widths are properties of the binning scheme rather than of the
final smeared bin curves.

This distinction is useful:

- **nominal widths** describe how the bins were defined,
- **effective widths** such as :math:`w_{68}` describe what the final
  bin curves actually look like after redshift uncertainty is included.

Binny also summarizes nominal widths collectively, for example through
their minimum, maximum, mean, and standard deviation, as well as an
**equidistant score** that quantifies how close the set of widths is to
being uniformly spaced.


Population statistics
---------------------

Population statistics describe how the total galaxy sample is divided
across tomographic bins.

These quantities do **not** come from the shapes of normalized bin
curves. Instead, they come from tomographic metadata returned during bin
construction.

Let :math:`f_i` denote the fraction of the total galaxy sample assigned
to bin :math:`i`. Then

.. math::

   \sum_i f_i = 1.

These fractions are the foundation for the population summaries
described below.


Galaxy fractions per bin
~~~~~~~~~~~~~~~~~~~~~~~~

The most direct population summary is the per-bin galaxy fraction
:math:`f_i`.

This tells us what share of the total sample is associated with each
bin. It is useful for assessing whether a given binning strategy
produces a balanced partition of the sample or instead concentrates more
galaxies into certain bins.

For example:

- **equipopulated binning** aims to make the :math:`f_i` values similar,
- **equidistant binning** generally produces more variation in
  :math:`f_i`, because it prioritizes redshift width rather than equal
  counts.

In practice, exact equality is not always achieved even for
equipopulated schemes, because redshift uncertainty can scatter galaxies
across bin boundaries and modify the final observed-bin populations.


Galaxy densities per bin
~~~~~~~~~~~~~~~~~~~~~~~~

If the total effective galaxy surface density of the sample is known,
say :math:`n_{\mathrm{tot}}` in galaxies per square arcminute, then the
effective density of bin :math:`i` is

.. math::

   n_i = f_i\,n_{\mathrm{tot}}.

This gives a more directly survey-relevant quantity than a pure
fraction. It tells us how much galaxy density is effectively available
in each bin for downstream analyses.

These per-bin densities are especially useful in forecasting, where
noise levels often depend on the effective source or lens density in
each tomographic slice.


Galaxy counts per bin
~~~~~~~~~~~~~~~~~~~~~

If one also specifies the survey area :math:`A`, measured in square
arcminutes, then an effective count can be assigned to each bin:

.. math::

   N_i = n_i A.

This converts per-bin surface densities into effective galaxy counts.

These counts are useful when connecting tomography summaries to survey
scale, shot-noise estimates, catalog expectations, or simple bookkeeping
in later pipeline stages.


Practical interpretation
------------------------

The various summaries described above do not all serve the same purpose.

Some quantities have a natural reference value:

- tail asymmetry is easiest to interpret relative to **1**,
- in-range fraction is easiest to interpret relative to **1**,
- second-peak strength is easiest to interpret relative to **0**.

Other quantities do not have a universal target:

- mean, median, and mode depend on the bin location,
- width summaries depend on the chosen science case,
- skewness and kurtosis depend on the detailed bin shape,
- nominal bin widths depend on the binning strategy itself.

Because of this, the most useful interpretation is usually comparative.
These summaries are most informative when used to compare:

- different binning schemes,
- different redshift uncertainty models,
- different numbers of tomographic bins,
- different survey assumptions.

In other words, the goal is usually not to declare a single number
“good” or “bad” in isolation, but to understand how tomographic choices
change the structure and population of the resulting bins.


How this connects to the examples
---------------------------------

This page introduces the concepts behind Binny's bin-summary statistics.

For worked examples showing how these quantities behave in practice,
including visualizations of bin centers, widths, and population
distributions, see the :doc:`Bin summaries examples <../examples/bin_summaries>`.

A useful way to think about the split is:

- this page explains **what the summary quantities mean**,
- the examples page shows **how they behave for real binning choices**.

Together they provide both the theoretical interpretation and the
practical diagnostics needed to assess tomographic bin quality.


Notes
-----

- Shape statistics are safe to compute from the bin curves themselves,
  even when those curves are individually normalized.
- Population statistics should be taken from tomography metadata rather
  than inferred from normalized curves.
- Different center definitions serve different purposes; no single
  choice is universally best.
- Robust percentile-based summaries are often easier to interpret than
  moment-based summaries when bin curves are skewed or mildly
  multimodal.
- In-range fraction and peak diagnostics are especially useful for
  understanding the impact of redshift uncertainty, leakage, and
  outliers.