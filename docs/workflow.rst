.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Workflow
===============

Binny provides a functional framework for constructing, combining,
and analyzing tomographic redshift distributions used in cosmology.

Tomographic bins can be generated from an underlying parent galaxy
redshift distribution :math:`n(z)`, supplied externally, or
constructed by combining existing samples.

The goal is to provide reproducible, flexible binning workflows
for cosmological forecasting, survey simulations, and data analysis.

Typical workflow
----------------

A typical workflow consists of five steps:

1. Define or load a parent redshift distribution.
2. Construct tomographic bin edges.
3. Select a redshift assignment model.
4. Build the tomographic distributions.
5. Inspect, compare, or combine the resulting bins.

These operations are exposed through independent functions rather
than requiring a persistent tomography object.

Parent redshift distribution
----------------------------

For model-generated tomography, the starting point is a parent
redshift distribution :math:`n(z)` defined on a redshift grid.

This distribution describes the underlying galaxy population before
tomographic selection is applied.

Binny supports several approaches:

- analytic distributions, such as the **Smail distribution**,
- distributions derived from luminosity functions,
- distributions inferred from mock catalogs,
- distributions incorporating observational selection effects,
- user-supplied redshift distributions.

The parent distribution can be evaluated and inspected independently
of the tomographic binning procedure.

Tomographic binning schemes
---------------------------

The next step is to define the tomographic bin boundaries.

Binny supports several binning strategies, including:

- **Equidistant binning:** bins with equal redshift widths.
- **Equipopulated binning:** bins containing approximately equal
  galaxy populations.
- **Custom bin edges:** boundaries supplied directly by the user.

Bin-edge construction is separate from the assignment model,
allowing different observational assumptions to be applied to
the same underlying binning scheme.

Redshift assignment models
--------------------------

A redshift assignment model determines how galaxies are assigned
to tomographic bins.

For spectroscopic samples, assignments can be based on measured
redshifts with optional observational uncertainties.

For photometric samples, the assignment model can incorporate:

- redshift scatter,
- systematic redshift offsets,
- redshift-dependent transformations,
- catastrophic outlier populations.

These effects modify the relationship between true redshift and
observed bin membership.

Separating bin-edge construction from redshift assignment makes
it possible to compare different uncertainty prescriptions while
retaining the same parent distribution and bin boundaries.

Building tomographic distributions
----------------------------------

Given a parent distribution, bin edges, and an assignment model,
Binny constructs the tomographic redshift distributions

.. math::

   n_i(z) = n(z) S_i(z),

where :math:`S_i(z)` represents the selection probability for
tomographic bin :math:`i`.

The resulting distributions describe the true-redshift populations
associated with each tomographic bin.

Depending on the chosen workflow, Binny can also return
normalization information, bin populations, and metadata.

The resulting arrays can be passed directly to cosmological
modelling and forecasting pipelines.

Predefined tomography
---------------------

Not every analysis requires constructing bins from a parent
distribution and an assignment model.

Binny also supports externally supplied tomographic distributions,
including those obtained from measurements, simulations, or
other analysis pipelines.

These distributions can be validated and prepared without
reapplying a redshift assignment model.

This is particularly useful when working with survey data
products or externally calibrated redshift distributions.

Sample composition
------------------

Tomographic distributions can also be combined to construct
new samples.

This allows workflows involving multiple galaxy populations,
survey selections, or independently generated tomographic bins.

When distributions are defined on different redshift grids,
they can be interpolated onto a common grid for combination.

Diagnostics and comparisons
---------------------------

After constructing tomography, Binny provides functions for
inspecting and comparing the resulting distributions.

Available diagnostics include:

- bin populations and normalization,
- mean redshifts and distribution widths,
- overlaps between tomographic bins,
- leakage between redshift intervals,
- comparisons between different tomographic samples.

These diagnostics are independent of the bin-construction
procedure and can also be applied to externally supplied bins.

Configuration vs Python usage
-----------------------------

Binny supports both direct Python workflows and
configuration-driven survey setups.

Users can:

- call individual functions to construct and analyze distributions,
- build tomography from structured specifications,
- load survey parameters from configuration files,
- use predefined survey presets.

The functional API allows individual operations to be reused
without requiring a particular workflow or object lifecycle.

The examples in :doc:`examples/index` demonstrate the available
approaches and provide executable illustrations of tomography
construction, diagnostics, and sample composition.
