.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Parent redshift distributions
====================================

This section describes how Binny models the underlying analytic parent
redshift distribution :math:`n(z)` used throughout tomographic analyses.

A parent :math:`n(z)` describes the redshift structure of a galaxy population
before any tomographic binning is applied. Binny provides both analytic parent
distribution models and workflows for calibrating survey-motivated
distributions from mock catalogs.

For examples showing how to evaluate parent distributions in practice, see the
corresponding examples pages.

Overview
--------

A tomographic analysis begins with a parent redshift distribution

.. math::

   n(z),

which describes how galaxies are distributed in redshift before any
tomographic selection is applied.

In Binny, the parent distribution serves as the starting point for
constructing tomographic bins, overlap diagnostics, photo-z modelling,
and downstream forecasting calculations.

Binny currently supports two complementary approaches:

- direct analytic parent-distribution models;
- calibration of survey-motivated parent distributions from mock catalogs.

In addition, Binny provides luminosity-function-weighted redshift
distributions that connect the parent :math:`n(z)` directly to a galaxy
luminosity function, survey magnitude limits, and cosmology-dependent
volume effects.

Parent distribution modelling
-----------------------------

The pages below describe the parent-distribution modelling approaches
implemented in Binny.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Analytic parent n(z) models
      :link: parent_nz
      :link-type: doc
      :class-card: sd-card-hover

   .. grid-item-card:: Parent n(z) calibration from mocks
      :link: nz_from_mocks
      :link-type: doc
      :class-card: sd-card-hover

   .. grid-item-card:: LF-weighted parent n(z)
      :link: nz_from_lf
      :link-type: doc
      :class-card: sd-card-hover


Detailed pages
--------------

.. toctree::
   :maxdepth: 1

   parent_nz
   nz_from_mocks
   nz_from_lf
   psf_selection
