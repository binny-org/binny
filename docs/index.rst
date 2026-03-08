Binny
=====

.. grid:: 2
   :gutter: 3

   .. grid-item::
      :class: hero-text

      **Binny** provides flexible, explicit, and well-tested tomographic
      binning tools for cosmology and related scientific workflows.

      It is designed for forecasting, inference, and data-processing
      pipelines, with an emphasis on clarity, reproducibility, and
      robust validation.

      Typical use cases include:

      - photometric-redshift tomography
      - spectroscopic-redshift binning
      - cosmological forecasting studies
      - survey-specific bin definitions

   .. grid-item::
      :class: sd-text-center hero-logo

      .. image:: _static/animations/binny_logo.gif
         :width: 220px
         :alt: Binny logo


Explore the documentation
-------------------------

.. grid:: 2
   :gutter: 4

   .. grid-item-card:: Theory
      :link: theory/index
      :link-type: doc
      :class-card: sd-card-hover

      Mathematical background and definitions behind tomographic
      binning in cosmology.

      Topics include:

      - redshift distributions :math:`n(z)`
      - photometric and spectroscopic selection
      - tomographic bin construction
      - overlap and leakage diagnostics

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc
      :class-card: sd-card-hover

      Practical demonstrations of Binny workflows.

      Includes:

      - building redshift distributions
      - constructing tomographic bins
      - survey configurations
      - diagnostics and summaries


Example gallery
---------------

The examples below illustrate common Binny workflows.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Parent redshift models
      :link: examples/nz_modelling
      :link-type: doc

      .. image:: _static/animations/parent_nz_model_sweep.gif
         :width: 100%
         :alt: Parent redshift models

      Construct analytic or empirical galaxy redshift distributions
      :math:`n(z)` and explore their properties.

   .. grid-item-card:: Calibrating redshift distributions
      :link: examples/nz_calibration
      :link-type: doc

      .. image:: _static/animations/smail_from_mock_calibration.gif
         :width: 100%
         :alt: Redshift calibration

      Infer redshift distribution parameters from simulations
      or mock galaxy catalogs.

   .. grid-item-card:: Photometric tomography
      :link: examples/photoz_bins
      :link-type: doc

      .. image:: _static/animations/tomo_photoz_example.gif
         :width: 100%
         :alt: Photometric bins

      Construct overlapping tomographic bins from photometric
      redshift estimates.

   .. grid-item-card:: Spectroscopic tomography
      :link: examples/specz_bins
      :link-type: doc

      .. image:: _static/animations/tomo_specz_example.gif
         :width: 100%
         :alt: Spectroscopic bins

      Define sharply bounded redshift bins suitable for
      spectroscopic surveys.

   .. grid-item-card:: Bin summaries
      :link: examples/bin_summaries
      :link-type: doc

      .. image:: _static/animations/tomo_bin_summaries.gif
         :width: 100%
         :alt: Bin summaries

      Summarize tomographic bins through effective redshifts,
      widths, number densities, and related aggregate statistics.

   .. grid-item-card:: Bin diagnostics
      :link: examples/bin_diagnostics
      :link-type: doc

      .. image:: _static/animations/tomo_bin_diagnostics.gif
         :width: 100%
         :alt: Bin diagnostics

      Inspect bin overlap, leakage, and statistical properties.

   .. grid-item-card:: Survey presets
      :link: examples/survey_presets
      :link-type: doc

      .. image:: _static/animations/lsst_preset_sweep.gif
         :width: 100%
         :alt: Survey presets

      Build tomographic bin configurations representative of
      real cosmological surveys.


Documentation
-------------

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