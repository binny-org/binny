.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Redshift uncertainties
=============================

This section describes how Binny models uncertainty in tomographic redshift
binning.

For a broader introduction to tomography and redshift-selection models, see
:doc:`../tomography`.

Binny supports two broad classes of redshift uncertainty treatment:

- :doc:`photoz` for **photometric-redshift uncertainties**, where bins are
  defined in observed redshift and mapped onto the true-redshift grid through a
  probabilistic assignment model;
- :doc:`specz` for **spectroscopic-redshift uncertainties**, where bins are
  defined in true redshift and may be modified by completeness losses,
  bin-to-bin response effects, catastrophic reassignment, or measurement
  scatter.

In both cases, the returned tomographic bins are evaluated on a common
true-redshift grid :math:`z`.

Overview
--------

Binny constructs tomographic bins by applying an effective redshift-selection
model to a parent redshift distribution :math:`n(z)`. Schematically,

.. math::

   n_i(z) = n(z)\, S_i(z),

where

- :math:`n(z)` is the parent redshift distribution,
- :math:`n_i(z)` is the returned tomographic bin for bin :math:`i`,
- :math:`S_i(z)` is the effective selection function.

The interpretation of :math:`S_i(z)` depends on the redshift model:

- in the **photo-z** case, :math:`S_i(z) = P(i \mid z)`, the probability of
  assigning an object at true redshift :math:`z` to observed bin :math:`i`;
- in the **spec-z** case, :math:`S_i(z)` is a true-redshift selection, possibly
  followed by an observed-bin response model.

Conceptually, the distinction is:

- **photo-z tomography** is probabilistic from the outset;
- **spec-z tomography** starts from deterministic true-redshift bins and then
  optionally adds observational response effects.


Uncertainty models
------------------

The pages below describe the uncertainty models implemented in Binny.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Photometric-redshift uncertainties
      :link: photoz
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: ../../_static/animations/pz_uncertainty_scatter.gif
         :width: 100%
         :alt: Photometric redshift uncertainty example

      Photometric redshift tomography assigns galaxies to bins
      probabilistically. These models capture scatter, bias, and
      catastrophic outliers in the mapping between observed and true
      redshift.

   .. grid-item-card:: Spectroscopic-redshift uncertainties
      :link: specz
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: ../../_static/animations/specz_uncertainty_scatter.gif
         :width: 100%
         :alt: Spectroscopic redshift uncertainty example

      Spectroscopic tomography begins with deterministic true-redshift
      bins and may include additional observational effects such as
      incompleteness, bin reassignment, or measurement scatter.


Detailed pages
--------------

.. toctree::
   :maxdepth: 1

   photoz
   specz