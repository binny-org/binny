.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Architecture
===================

Binny is a modular Python framework for modeling, constructing,
calibrating, and analyzing galaxy redshift distributions and
tomographic samples in cosmology.

Its architecture separates galaxy population modeling, binning,
tomographic construction, observational calibration, statistical
analysis, and survey-specific configurations into distinct
components.

These components can be used independently or combined through
a unified functional API, allowing users to construct workflows
appropriate to their scientific applications without imposing
a particular survey, observational model, or analysis strategy.

This page introduces the main components of Binny, explains
their responsibilities, and describes how they work together.


The six pillars of Binny
------------------------

Binny is organized around six primary public API namespaces.

Each namespace addresses a distinct scientific or computational
task, from defining a galaxy population to constructing and
analyzing tomographic redshift distributions.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Namespace
     - Responsibility
     - Functionality
   * - ``binny.nz``
     - Galaxy population modeling
     - Defines, loads, and samples parent redshift distributions,
       including analytic, tabulated, and luminosity-function-based
       models.
   * - ``binny.binning``
     - Binning and galaxy assignment
     - Constructs and resolves bin boundaries, defines assignment
       and selection prescriptions, and computes binned redshift
       distributions.
   * - ``binny.tomography``
     - Tomographic construction and representation
     - Combines galaxy populations, binning prescriptions, and
       assignment models into a unified tomographic representation.
       Also supports externally supplied tomographic distributions.
   * - ``binny.calibration``
     - Observational calibration and fitting
     - Fits photometric redshift-error models and calibrates galaxy
       redshift distributions, number densities, survey depth,
       and observational selection effects from mock catalogs.
   * - ``binny.analysis``
     - Statistical analysis and sample manipulation
     - Computes distribution statistics, population properties,
       overlap, leakage, correlations, bin combinations, and
       comparisons, and supports sample composition.
   * - ``binny.surveys``
     - Survey configurations and presets
     - Provides survey-specific assumptions, configuration-driven
       tomography construction, and survey area and number-density
       conversions.

The six namespaces provide distinct but complementary
functionality.

A workflow does not need to use all six components.
For example, users may construct a parent redshift distribution
without performing tomography, calibrate an observational model
without constructing bins, or analyze externally supplied
tomographic distributions without using a Binny binning
prescription.


Galaxy population modeling: ``binny.nz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.nz`` namespace provides tools for defining and
working with galaxy redshift distributions.

A parent redshift distribution, :math:`n(z)`, describes the
underlying galaxy population before tomographic selection.

Supported functionality includes:

* Analytic redshift distribution models.
* Tabulated and externally supplied redshift distributions.
* Luminosity-function-based redshift distributions.
* Registration and retrieval of distribution models.
* Sampling individual galaxy redshifts from a distribution.
* Construction of mock redshift catalogs, optionally including
  observational redshifts and galaxy weights.

The parent distribution is independent of the binning and
tomography construction procedures.

The same galaxy population can therefore be reused across
different binning schemes, observational assumptions, and
survey configurations.

Conversely, different galaxy population models can be compared
while keeping the remaining components of the analysis fixed.


Binning and galaxy assignment: ``binny.binning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.binning`` namespace defines how galaxy populations
are divided into bins.

Binning involves two conceptually distinct operations:

* Determining where the bin boundaries lie.
* Determining how galaxies are assigned to those bins.

The first operation defines the binning geometry.
The second defines the selection or assignment prescription.

Supported binning schemes include equidistant, equipopulated,
equal-information, logarithmic, geometric, power-law,
Chebyshev, sinh, and generalized binning.

Users can also supply explicit bin boundaries, register custom
binning schemes, restrict the binning range, or combine
multiple schemes into segmented binning configurations.

Galaxy assignment can incorporate:

* Deterministic redshift selection.
* Spectroscopic redshift uncertainties.
* Photometric redshift uncertainties.
* Systematic redshift offsets.
* Catastrophic outlier populations.
* Response matrices.
* User-defined assignment functions.

Separating bin boundaries from galaxy assignment allows users
to investigate observational uncertainties without modifying
the underlying binning geometry.

Similarly, alternative binning strategies can be evaluated
while keeping the observational model fixed.

The binning functionality can be used directly or through
the higher-level tomography construction interface.


Tomographic construction: ``binny.tomography``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.tomography`` namespace provides the construction
and representation of tomographic redshift distributions.

Given a parent redshift distribution :math:`n(z)` and a
selection probability :math:`S_i(z)`, the unnormalized
tomographic redshift distribution for bin :math:`i` is

.. math::

   n_i(z) = n(z) S_i(z).

The selection function describes the probability that a galaxy
at true redshift :math:`z` is assigned to tomographic bin
:math:`i`.

For an observational redshift :math:`z_{\mathrm{obs}}` with
conditional probability density
:math:`p(z_{\mathrm{obs}} \mid z)`, selection into a bin with
boundaries :math:`a_i` and :math:`b_i` can be expressed as

.. math::

   S_i(z) =
   \int_{a_i}^{b_i}
   p(z_{\mathrm{obs}} \mid z)
   \,\mathrm{d}z_{\mathrm{obs}}.

This formulation separates the underlying galaxy population
from the observational prescription used to assign galaxies
to tomographic bins.

The resulting bin distributions may optionally be normalized
according to the requirements of the analysis.

The main construction interfaces are:

* ``build_tomography`` for direct functional construction.
* ``build_tomography_from_spec`` for construction from a
  structured tomography specification.
* ``build_predefined_tomography`` for constructing tomography
  from externally supplied bin distributions.
* ``load_predefined_bins`` for loading externally supplied
  tomographic distributions.

These construction pathways use a common ``Tomography``
representation containing the redshift grid, parent
distribution, tomographic bin distributions, and associated
information where available.

Externally supplied tomographic distributions do not need
to be reconstructed using a Binny binning or assignment
prescription.

The responsibility of this namespace is to construct and
represent tomography.

Statistical analysis, distribution comparisons, and sample
composition are provided separately by ``binny.analysis``.


Observational calibration: ``binny.calibration``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.calibration`` namespace connects observational
or simulated galaxy samples to the models used in redshift
distribution and tomography construction.

Rather than requiring all model parameters to be specified
manually, calibration allows relevant quantities to be
estimated from galaxy catalogs.

Supported functionality includes:

* Fitting Smail-type redshift distribution parameters.
* Calibrating magnitude-dependent galaxy redshift distributions.
* Calibrating galaxy number densities as a function of survey
  depth or limiting magnitude.
* Fitting photometric redshift-error distributions, including
  core and outlier populations.
* Converting fitted photometric redshift-error parameters into
  Binny-compatible model parameters.
* Modeling PSF-dependent galaxy selection.
* Constructing weighted redshift distributions from mock
  galaxy catalogs.
* Estimating selected galaxy number densities.

Calibration and model evaluation are distinct operations.

For example, a redshift distribution model can be evaluated
directly through ``binny.nz``, while its parameters can be
estimated from a mock catalog through ``binny.calibration``.

Similarly, a photometric redshift assignment model can be
evaluated through ``binny.binning``, while its uncertainty
parameters can be estimated through ``binny.calibration``.

Calibrated distributions and model parameters can subsequently
be used as inputs to tomography construction.

Calibration is optional.
Users with externally determined model parameters or redshift
distributions can use those inputs directly.


Statistical analysis: ``binny.analysis``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.analysis`` namespace provides tools for examining,
comparing, and manipulating redshift distributions and
tomographic samples.

These operations are independent of how the distributions
were originally constructed.

The analysis tools can therefore be applied to Binny-generated
tomography, observational data products, simulations, or
externally supplied tomographic distributions.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Capability
     - Functionality
   * - Distribution statistics
     - Compute means, medians, modes, moments, quantiles,
       distribution widths, bin centers, and peak diagnostics.
   * - Population statistics
     - Compute galaxy fractions, number densities, galaxy counts,
       comoving number densities, and in-range population
       fractions.
   * - Distribution comparisons
     - Quantify similarities and differences between
       redshift distributions and tomographic samples.
   * - Overlap and leakage
     - Examine the relationships between tomographic bins
       and quantify population leakage between redshift
       intervals.
   * - Correlations
     - Evaluate statistical relationships between distributions
       using correlation and overlap metrics.
   * - Bin combinations
     - Construct and filter bin pairs and higher-order bin
       combinations using distribution properties, overlap,
       and other selection criteria.
   * - Sample composition
     - Combine parent distributions and tomographic samples,
       align different redshift grids, interpolate distributions,
       and construct sample combinations.

These capabilities support both exploratory analyses and
the preparation of tomographic inputs for downstream
cosmological calculations.

Analysis is not restricted to the final stage of a workflow.

For example, overlap and population statistics can be used
to assess alternative binning schemes, compare observational
models, or evaluate the consequences of different calibration
prescriptions.


Survey configurations: ``binny.surveys``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``binny.surveys`` namespace provides configuration-driven
workflows and survey-specific assumptions.

Survey configurations can describe:

* Parent redshift distribution models.
* Lens and source galaxy populations.
* Survey years, scenarios, and sample definitions.
* Tomographic binning prescriptions.
* Photometric and spectroscopic redshift uncertainties.
* Galaxy number densities.
* Survey areas.
* Sample-dependent properties.

The ``build_survey_tomography`` interface uses these
configurations to construct tomographic samples through
the underlying Binny framework.

The namespace also provides utilities for inspecting survey
configurations and converting between survey area, sky fraction,
solid angle, and angular number-density units.

Survey presets do not constitute a separate numerical
implementation of tomography.

Instead, they provide a structured way to define inputs to
the same galaxy population, binning, and tomography
construction machinery available through the functional API.

Users may therefore choose between predefined survey
configurations, custom configurations, structured tomography
specifications, and direct Python construction.


Supporting namespaces
---------------------

In addition to the six primary API namespaces, Binny provides
specialized and shared functionality through supporting
namespaces.

These modules remain independently accessible and can be
used directly when their functionality is needed outside
the primary workflow.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Namespace
     - Responsibility
     - Functionality
   * - ``binny.axes``
     - Coordinate grids and bin boundaries
     - Provides coordinate grid generators and numerical
       implementations of bin-edge construction schemes.
   * - ``binny.correlations``
     - Correlation and combination utilities
     - Provides distribution metrics, scoring functions,
       bin-combination filtering, and pair and tuple
       construction utilities.
   * - ``binny.cosmology``
     - Cosmological calculations
     - Provides cosmology-dependent distances, volume weights,
       and redshift-related quantities used by supported
       distribution and binning models.
   * - ``binny.utils``
     - Shared utilities
     - Provides numerical integration and normalization,
       validation, interpolation-related utilities, metadata,
       file loading, and visualization utilities.

Selected functionality from the supporting namespaces is
also exposed through the primary API namespaces where
appropriate.

For example, bin-edge generators are accessible through
``binny.binning``, while correlation and bin-combination
utilities are accessible through ``binny.analysis``.

These interfaces reuse the underlying implementations
rather than introducing separate numerical prescriptions.


How the components work together
--------------------------------

The components of Binny are designed to support flexible
scientific workflows rather than enforce a single sequence
of operations.

A typical tomography construction workflow can be represented
as follows:

.. code-block:: text

                     Galaxy catalogs
                           |
                           v
                   binny.calibration
                           |
                  Calibrated parameters
                           |
                           v
       binny.nz                         binny.binning
       Parent n(z)                 Boundaries and selection
            |                                  |
            +----------------+-----------------+
                             |
                             v
                     binny.tomography
                     Tomographic samples
                             |
                             v
                       binny.analysis
               Statistics and sample manipulation

The diagram represents conceptual relationships between
the components rather than a strict sequence of function
calls.

Calibration is optional, and parent distributions and
assignment parameters can be supplied directly.

The tomography construction interfaces coordinate the
necessary binning operations internally, while the
individual components remain independently accessible.

Survey configurations provide an additional entry point
to this workflow:

.. code-block:: text

                    binny.surveys
                  Survey configuration
                           |
                           v
              Population and binning inputs
                           |
                           v
                   binny.tomography
                   Tomographic samples
                           |
                           v
                     binny.analysis

Alternatively, externally supplied tomographic distributions
can bypass parent distribution modeling and galaxy assignment:

.. code-block:: text

             External tomographic distributions
                           |
                           v
                   binny.tomography
                           |
                           v
                     binny.analysis

These workflows produce a common tomographic representation,
allowing the same analysis tools to be applied regardless
of the origin of the distributions.


Choosing a workflow
-------------------

The appropriate entry point depends on the available inputs
and the scientific objective.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Scientific task
     - Recommended namespace
   * - Construct an analytic or tabulated parent redshift
       distribution.
     - ``binny.nz``
   * - Generate a mock galaxy redshift catalog.
     - ``binny.nz``
   * - Compare alternative binning schemes.
     - ``binny.binning`` and ``binny.analysis``
   * - Model photometric or spectroscopic redshift
       uncertainties.
     - ``binny.binning``
   * - Estimate photometric redshift-error parameters
       from a galaxy catalog.
     - ``binny.calibration``
   * - Calibrate redshift distributions and galaxy number
       densities as a function of survey depth.
     - ``binny.calibration``
   * - Construct tomography from a parent population and
       an assignment model.
     - ``binny.tomography``
   * - Load externally supplied tomographic distributions.
     - ``binny.tomography``
   * - Evaluate tomographic populations, overlap,
       leakage, and distribution statistics.
     - ``binny.analysis``
   * - Combine or compare multiple tomographic samples.
     - ``binny.analysis``
   * - Select bin pairs or higher-order bin combinations
       for a cosmological analysis.
     - ``binny.analysis``
   * - Construct tomography using survey-specific
       assumptions.
     - ``binny.surveys``


Design principles
-----------------

The organization of Binny is based on several principles.

**Separation of responsibilities**

Galaxy population modeling, binning, tomography construction,
calibration, and statistical analysis are distinct operations
with different scientific responsibilities.

**Independent components**

Individual components can be used outside a complete
tomographic workflow, allowing users to adopt only the
functionality required by their analysis.

**Functional construction**

The primary construction interfaces accept explicit inputs
and return results without requiring stateful builder
objects or a prescribed sequence of configuration steps.

**Reusable numerical implementations**

Direct construction, structured specifications, and survey
presets share the underlying numerical machinery.

**Observational flexibility**

Different galaxy populations, redshift-error models,
selection functions, and survey assumptions can be combined
without requiring changes to the overall tomography
framework.

**Support for external data**

Externally supplied redshift distributions and tomographic
samples can be incorporated without requiring them to be
reconstructed through Binny.

**Interoperability**

Redshift distributions, tomographic samples, calibrated
model parameters, and statistical diagnostics can be used
as inputs to external cosmological modeling, simulation,
forecasting, and inference frameworks.


Further reading
---------------

See :doc:`/examples/index` for worked examples illustrating
the different components and construction workflows.

See :doc:`/api/binny` for the complete API reference,
including the primary and supporting namespaces.