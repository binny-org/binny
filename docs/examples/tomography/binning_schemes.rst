
Binning schemes
===============

Binny supports several prescriptions for constructing tomographic bin
edges. All prescriptions use a unified ``bins`` dictionary, which
separates the binning configuration from the parent redshift distribution
and the redshift uncertainty model.

A bin specification selects exactly one of three modes:

- Explicit edges, using ``edges``.
- Generated edges, using ``scheme`` and ``n_bins``.
- Mixed binning, using ``segments``.

The central resolver is
:func:`binny.binning.edges.resolve_bin_edges`.

Explicit bin edges
------------------

When the bin boundaries are already known, they can be supplied directly.

.. code-block:: python

   bins = {
       "edges": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
   }

The edges must form a strictly increasing sequence. This mode does not
accept a generating scheme, number of bins, or separate binning range.

Generated bin edges
-------------------

Generated binning uses a named scheme and a requested number of bins.

.. code-block:: python

   bins = {
       "scheme": "equidistant",
       "n_bins": 5,
       "range": [0.2, 1.2],
   }

The optional ``range`` restricts the interval over which edges are
constructed. If omitted, the supplied redshift grid determines the
interval.

Scheme-specific parameters belong in the nested ``params`` dictionary.

Built-in schemes
~~~~~~~~~~~~~~~~

The following schemes are registered by default.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Scheme
     - Aliases
     - Description
   * - ``equidistant``
     - ``eq``, ``linear``
     - Equal-width redshift intervals.
   * - ``equal_number``
     - ``equipopulated``, ``en``
     - Equal integrated population per bin.
   * - ``generalized``
     - None
     - Equal integrated generalized weight per bin.

Equidistant binning
~~~~~~~~~~~~~~~~~~~

Equidistant binning divides the requested interval into equal-width
redshift bins.

For an interval from :math:`z_{\min}` to :math:`z_{\max}`, the edges are

.. math::

   z_i = z_{\min}
   + \frac{i}{N}(z_{\max}-z_{\min}),
   \qquad i=0,\ldots,N.

For example:

.. code-block:: python

   bins = {
       "scheme": "equidistant",
       "n_bins": 5,
       "range": [0.2, 1.2],
   }

Equal-number binning
~~~~~~~~~~~~~~~~~~~~

Equal-number binning selects boundaries such that each bin contains the
same integrated population.

For a parent distribution :math:`n(z)`, the boundaries satisfy

.. math::

   \int_{z_i}^{z_{i+1}} n(z)\,\mathrm{d}z
   =
   \frac{1}{N}
   \int_{z_{\min}}^{z_{\max}} n(z)\,\mathrm{d}z.

For example:

.. code-block:: python

   bins = {
       "scheme": "equal_number",
       "n_bins": 5,
       "range": [0.2, 1.2],
   }

The alias ``equipopulated`` is equivalent to ``equal_number``.

The resolver can also use a separate distribution for computing
equal-number edges through ``equal_number_axis`` and
``equal_number_weights``.

Generalized binning
-------------------

Generalized binning constructs boundaries by integrating a weight
function that depends on the parent redshift distribution and,
optionally, the comoving radial distance.

The generalized weight is

.. math::

   w(z) =
   [n(z)]^\alpha
   \left(\frac{\mathrm{d}\chi}{\mathrm{d}z}\right)^\beta,

where :math:`\chi(z)` is the comoving radial distance and
:math:`\alpha` and :math:`\beta` are configurable exponents.

The edges satisfy

.. math::

   \int_{z_i}^{z_{i+1}} w(z)\,\mathrm{d}z
   =
   \frac{1}{N}
   \int_{z_{\min}}^{z_{\max}} w(z)\,\mathrm{d}z.

This prescription contains several familiar binning strategies as
special cases.

.. list-table::
   :header-rows: 1

   * - :math:`\alpha`
     - :math:`\beta`
     - Interpretation
   * - 0
     - 0
     - Equidistant redshift binning
   * - 1
     - 0
     - Equal-number binning
   * - 0
     - 1
     - Equal comoving-distance binning

Intermediate exponent values allow bin boundaries to interpolate between
these prescriptions.

For example, generalized binning with :math:`\alpha=0.5` and
:math:`\beta=0` can be configured as follows:

.. code-block:: python

   bins = {
       "scheme": "generalized",
       "n_bins": 5,
       "params": {
           "alpha": 0.5,
           "beta": 0.0,
       },
   }

When :math:`\beta=0`, comoving-distance information is unnecessary.

For nonzero :math:`\beta`, the underlying
:func:`binny.axes.bin_edges.generalized_tomographic_edges`
function requires a comoving-distance array evaluated on the same
redshift grid.

.. code-block:: python

   from binny.axes.bin_edges import generalized_tomographic_edges

   edges = generalized_tomographic_edges(
       z,
       nz,
       n_bins=5,
       alpha=0.5,
       beta=1.0,
       comoving_distance=chi,
   )

Here ``z``, ``nz``, and ``chi`` are one-dimensional arrays, with
``chi`` containing strictly increasing comoving radial distances.

The current ``resolve_bin_edges`` interface does not expose a
``comoving_distance`` argument. Therefore, generalized configurations
with nonzero ``beta`` should use the lower-level function directly.

Mixed binning
-------------

Mixed binning combines different binning methods across adjacent
redshift intervals.

Each segment specifies a scheme, a number of bins, and its redshift
boundaries.

For example:

.. code-block:: python

   bins = {
       "segments": [
           {
               "scheme": "equidistant",
               "n_bins": 2,
               "z_min": 0.2,
               "z_max": 0.6,
           },
           {
               "scheme": "equal_number",
               "n_bins": 3,
               "z_min": 0.6,
               "z_max": 1.2,
           },
       ],
   }

The resulting edges form one increasing array. Shared boundaries
between adjacent segments are included only once.

The total number of bins is the sum of the segment counts.

Mixed binning uses the underlying
:func:`binny.axes.mixed_edges.mixed_edges` machinery.

For equal-number segments, the parent distribution is used as the
population proxy unless a separate photo-z distribution is provided
through ``z_ph`` and ``nz_ph``.

Resolving bin specifications
----------------------------

The same resolver accepts explicit, generated, and mixed bin
specifications.

.. code-block:: python

   import numpy as np

   from binny.binning.edges import resolve_bin_edges

   z = np.linspace(0.0, 3.0, 501)
   nz = z**2 * np.exp(-z / 0.5)

   bins = {
       "scheme": "generalized",
       "n_bins": 5,
       "params": {
           "alpha": 0.5,
           "beta": 0.0,
       },
   }

   edges = resolve_bin_edges(
       axis=z,
       density=nz,
       bins=bins,
   )

   print(edges)

The returned array contains ``n_bins + 1`` edges.

The resolver validates the specification and dispatches it to the
appropriate binning implementation.

Survey YAML configurations
--------------------------

The same ``bins`` structure is used in survey presets.

For example:

.. code-block:: yaml

   tomography:
     - role: lens
       kind: photoz

       nz:
         model: smail
         params:
           z0: 0.5
           alpha: 2.0
           beta: 1.0

       bins:
         scheme: generalized
         n_bins: 5
         params:
           alpha: 0.5
           beta: 0.0

       uncertainties:
         scatter_scale: 0.05
         mean_offset: 0.0

       normalize_bins: true

The ``nz`` parameters describe the parent distribution, while the
parameters nested under ``bins`` describe the binning prescription.

The uncertainty model is configured independently.

Custom binning schemes
----------------------

Additional binning schemes can be registered through
:func:`binny.binning.scheme_registry.register_binning_scheme`.

A registered scheme receives the redshift grid, parent distribution,
and requested number of bins. It must return the corresponding
numerical edges.

For example:

.. code-block:: python

   import numpy as np

   from binny.binning.scheme_registry import register_binning_scheme
   from binny.binning.edges import resolve_bin_edges

   @register_binning_scheme("quadratic_spacing")
   def quadratic_spacing(z, nz, n_bins):
       t = np.linspace(0.0, 1.0, n_bins + 1)
       return z[0] + (z[-1] - z[0]) * t**2

   z = np.linspace(0.0, 3.0, 501)
   nz = z**2 * np.exp(-z / 0.5)

   edges = resolve_bin_edges(
       axis=z,
       density=nz,
       bins={
           "scheme": "quadratic_spacing",
           "n_bins": 5,
       },
   )

Registered schemes are available by name through the same resolver.

Scheme names and aliases are case-insensitive. Duplicate registered
names are rejected.

The registry also provides
:func:`binny.binning.scheme_registry.get_binning_scheme`
and
:func:`binny.binning.scheme_registry.available_binning_schemes`.

See also
--------

- :doc:`photoz_bins`
- :doc:`specz_bins`
- :doc:`../../theory/tomography`
- :mod:`binny.binning.edges`
- :mod:`binny.axes.bin_edges`
