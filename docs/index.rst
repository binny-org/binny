Binny
=====

.. image:: _static/assets/logo.png
   :alt: Binny logo
   :width: 150px
   :align: right

**Binny** is a Python library providing flexible, explicit, and well-tested
tomographic binning algorithms for cosmology and related scientific workflows.

It is designed to integrate easily into forecasting, inference, and
data-processing pipelines, with an emphasis on clarity, reproducibility,
and robust validation.


Overview
--------

Binny provides tools for constructing and validating tomographic binning
schemes commonly used in cosmology and large-scale structure analyses.

The package focuses on explicit bin definitions, reproducible binning
strategies, and diagnostics that help compare different tomographic choices.

Typical use cases include:

- photometric-redshift tomography
- spectroscopic-redshift binning
- cosmological forecasting studies
- survey-specific bin definitions


Example gallery
---------------

The examples below illustrate common Binny workflows.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Parent redshift models
      :link: examples/nz_modelling
      :link-type: doc

      Construct analytic or empirical galaxy redshift distributions
      :math:`n(z)` and explore their properties.

   .. grid-item-card:: Calibrating redshift distributions
      :link: examples/nz_calibration
      :link-type: doc

      Infer redshift distribution parameters from simulations
      or mock galaxy catalogs.

   .. grid-item-card:: Photometric tomography
      :link: examples/photoz_bins
      :link-type: doc

      Construct overlapping tomographic bins from photometric
      redshift estimates.

   .. grid-item-card:: Spectroscopic binning
      :link: examples/specz_bins
      :link-type: doc

      Define sharply bounded redshift bins suitable for
      spectroscopic surveys.

   .. grid-item-card:: Bin diagnostics
      :link: examples/bin_diagnostics
      :link-type: doc

      Inspect bin overlap, leakage, and statistical properties.

   .. grid-item-card:: Survey presets
      :link: examples/survey_presets
      :link-type: doc

      Build tomographic bin configurations representative of
      real cosmological surveys.


Documentation
-------------

The documentation is organized into three main parts:

- :doc:`theory/index` — theoretical background and mathematical definitions
- :doc:`workflow` — recommended analysis workflow using Binny
- :doc:`examples/index` — practical examples and tutorials


.. toctree::
   :maxdepth: 1
   :caption: Contents

   theory/index
   workflow
   examples/index
   installation
   citation
   contributing
   api/index