.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Workflow
===============

Binny constructs tomographic redshift bins from an underlying parent
galaxy redshift distribution :math:`n(z)`.

The goal is to define reproducible binning schemes that can be used in
forecasting pipelines, cosmological analyses, or survey simulations.

Typical workflow
----------------

Constructing tomographic bins with Binny usually involves four steps:

1. Define a parent redshift distribution.
2. Choose a tomographic binning scheme.
3. Specify redshift uncertainty models.
4. Build the tomographic bins and inspect the results.

Each of these ingredients is described below.

Parent redshift distribution
----------------------------

All tomographic bins are derived from a **parent redshift distribution**
:math:`n(z)` defined on a redshift grid.

This distribution represents the underlying galaxy population before
any binning is applied.

Binny provides several ways to obtain a parent distribution:

- analytic models (for example the **Smail distribution**),
- empirical fits inferred from mock catalogs,
- or user-supplied redshift distributions.

The parent distribution is always evaluated on a shared redshift grid,
which ensures that all bins remain comparable and consistent.

Tomographic binning schemes
---------------------------

The next step is to divide the parent distribution into tomographic bins.

Binny supports several binning strategies commonly used in cosmology,
including:

- **Equidistant binning**
  bins are spaced evenly in redshift.

- **Equipopulated binning**
  each bin contains approximately the same number of galaxies.

These schemes define the **ideal bin boundaries** before any observational
uncertainties are applied.

Redshift uncertainty models
---------------------------

In real surveys the measured redshifts are not exact.

Photometric surveys typically include several sources of uncertainty,
such as:

- redshift scatter,
- systematic bias,
- catastrophic outliers.

Spectroscopic samples usually have much smaller uncertainties but may
still include small measurement errors or incomplete sampling.

Binny allows these effects to be modeled explicitly so that the resulting
bins reflect realistic observational conditions.

Building tomographic bins
-------------------------

Once the parent distribution, binning scheme, and uncertainty model are
defined, Binny constructs the tomographic bins.

The result is a set of bin distributions

.. math::

    n_i(z)

representing the redshift distribution of galaxies in each tomographic
bin.

These distributions can then be used directly in cosmological calculations
such as weak lensing forecasts, galaxy clustering analyses, or survey
design studies.

Configuration vs Python usage
-----------------------------

Binny can be used either through direct Python calls or through
configuration-driven workflows.

Users may:

- construct distributions and bins directly in Python,
- load survey parameters from configuration files (such as YAML),
- or use predefined survey presets.

The examples in :doc:`examples/index` demonstrate several typical
binning setups and provide executable scripts illustrating the workflow.