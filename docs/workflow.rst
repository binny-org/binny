.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Workflow
===============

Binny is a modular Python framework for constructing, manipulating,
and analyzing tomographic redshift distributions in cosmology.

Its architecture separates the definition of galaxy populations,
coordinate systems, binning prescriptions, and tomographic
distributions into independent components. These components can be
used individually or combined through a unified functional API.

This page describes how Binny is organized, what each module is
responsible for, and how the modules work together in a typical
cosmological analysis.

Package organization
--------------------

Binny is organized into modules with distinct responsibilities.
The first four form the core tomography workflow, while the
remaining modules provide analysis, cosmological utilities,
and configuration-driven workflows.

.. list-table::
   :header-rows: 1
   :widths: 20 31 49

   * - Module
     - Responsibility
     - Functionality
   * - ``binny.nz``
     - Galaxy redshift distributions
     - Defines or loads the parent :math:`n(z)`, including analytic
       models, luminosity-function-based distributions, and
       observational selection effects.
   * - ``binny.axes``
     - Coordinates and bin boundaries
     - Constructs redshift grids and bin edges using predefined
       schemes, custom boundaries, or mixed binning strategies.
   * - ``binny.binning``
     - Binning and galaxy assignment
     - Resolves binning specifications and applies spectroscopic,
       photometric, or custom assignment prescriptions, including
       redshift uncertainties and outlier populations.
   * - ``binny.tomography``
     - Tomographic construction and manipulation
     - Combines the parent distribution, bin edges, and assignment
       model into tomographic distributions. Also supports predefined
       tomography, sample composition, and statistical diagnostics.
   * - ``binny.correlations``
     - Distribution comparisons and correlations
     - Provides measures for comparing distributions and
       characterizing statistical relationships.
   * - ``binny.cosmology``
     - Cosmological calculations
     - Provides cosmology-dependent quantities used by supported
       redshift-distribution and binning prescriptions.
   * - ``binny.surveys``
     - Survey configurations and presets
     - Loads survey-specific assumptions and constructs tomography
       through predefined or user-supplied configurations.
   * - ``binny.utils``
     - Shared utilities
     - Provides numerical, validation, normalization, and
       visualization utilities used throughout the package.

Core workflow
-------------

The standard workflow follows four conceptual stages:

.. code-block:: text

   Parent redshift distribution
            binny.nz
                |
                v
   Coordinates and bin boundaries
            binny.axes
                |
                v
   Binning and assignment models
           binny.binning
                |
                v
   Tomographic redshift distributions
          binny.tomography
                |
                v
   Statistics, diagnostics, comparisons,
   and sample composition

The diagram represents the conceptual workflow rather than a strict
sequence of function calls. The ``build_tomography`` interface can
coordinate bin-edge construction and assignment internally, while
individual components remain independently accessible.

1. Parent redshift distributions: ``binny.nz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The starting point is a parent galaxy redshift distribution
:math:`n(z)`, describing the underlying population before
tomographic selection.

Binny supports analytic distributions, luminosity-function-based
models, observational selection effects, and externally supplied
distributions.

The parent distribution is independent of the tomography
construction procedure and can be reused across different
binning schemes and observational assumptions.

2. Coordinates and bin edges: ``binny.axes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The axes module defines the coordinate grids and boundaries
on which tomographic bins are constructed.

Supported schemes include equidistant, equipopulated,
equal-information, logarithmic, geometric, power-law,
Chebyshev, sinh, and generalized binning.

Users can also provide explicit bin edges or combine multiple
schemes into segmented binning configurations.

The central responsibility of this module is to determine
**where the bin boundaries lie**, without specifying how
galaxies are assigned to those bins.

3. Binning and assignment: ``binny.binning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The binning module connects bin boundaries with the
observational prescription used to assign galaxies.

For spectroscopic samples, assignments may be deterministic
or incorporate observational redshift uncertainties.

For photometric samples, the assignment model can include
redshift scatter, systematic offsets, and catastrophic
outlier populations.

User-defined assignment functions can also be supplied.

This separation allows the same parent distribution and
bin boundaries to be used with different observational models.

4. Tomography construction: ``binny.tomography``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tomography module brings the preceding components together.

Given a parent distribution :math:`n(z)` and a selection
probability :math:`S_i(z)`, the tomographic redshift distribution
for bin :math:`i` is

.. math::

   n_i(z) = n(z) S_i(z).

The selection function represents the probability that a
galaxy at true redshift :math:`z` belongs to bin :math:`i`.

The main entry point, ``build_tomography``, coordinates the
construction and returns a ``Tomography`` result containing
the tomographic distributions and associated information.

The module also supports externally supplied, predefined
tomographic distributions without reapplying a binning or
assignment prescription.

Analysis and sample manipulation
--------------------------------

After tomography is constructed, Binny provides additional
functionality for examining and manipulating the results.

These operations are independent of the construction method
and can also be applied to externally supplied tomography.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Capability
     - Module
     - Purpose
   * - Statistics
     - ``binny.tomography``
     - Compute bin populations, mean redshifts, distribution
       widths, moments, quantiles, and normalization statistics.
   * - Diagnostics
     - ``binny.tomography``
     - Examine bin overlap, leakage, and differences between
       tomographic samples.
   * - Correlations
     - ``binny.correlations``
     - Quantify statistical relationships and correlations
       between distributions.
   * - Sample composition
     - ``binny.tomography``
     - Combine samples, interpolate between redshift grids,
       and construct new tomographic populations.
   * - Photo-z calibration
     - ``binny.binning``
     - Fit redshift-error prescriptions and convert fitted
       parameters into models suitable for tomography.

These tools can be used separately or together, depending
on the requirements of a particular analysis.

Survey presets and configurations
----------------------------------

The ``binny.surveys`` module provides a convenient alternative
to specifying every component manually.

Survey configurations collect assumptions about the parent
redshift distribution, binning scheme, redshift-error model,
and other survey-dependent parameters.

The ``build_survey_tomography`` function uses these
configurations to construct tomography through the same
underlying numerical framework.

Conceptually, survey presets provide inputs to the core
workflow rather than defining a separate implementation:

.. code-block:: text

             Survey configuration
                binny.surveys
                      |
                      v
          Parent n(z) + binning settings
                      |
                      v
               build_tomography
                      |
                      v
                  Tomography
                      |
                      v
           Statistics and diagnostics

Users can therefore choose between direct Python construction,
structured tomography specifications, and survey presets
without changing the underlying mathematical prescriptions.

See :doc:`examples/index` for executable examples of the
different workflows.
