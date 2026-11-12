.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Binning schemes
======================

Binny supports several prescriptions for constructing tomographic bin
edges. The binning prescription is independent of the parent redshift
distribution and the redshift uncertainty model.

A binning scheme determines where the bin boundaries are placed.
The tomography kind (for example, ``photoz`` or ``specz``) determines
how objects are assigned to those bins.

These are independent choices: the same binning scheme can be used
with different tomography kinds.

For practical configurations and executable examples, see
:doc:`../examples/tomography/binning_schemes`.


Redshift grid and binning interval
----------------------------------

.. note::

   **The redshift grid and the binning interval are different things.**

   The input redshift grid (``z``, or ``axis`` in the edge resolver)
   specifies the redshift values at which the parent distribution
   is evaluated.

   By default, Binny constructs bin edges over the full range of
   the supplied grid. To restrict binning to a smaller interval,
   specify ``range`` inside the ``bins`` dictionary.

   Alternatively, use ``edges`` to provide the exact bin boundaries
   without selecting a generating scheme.

   **Restricting the binning interval does not redefine the input
   redshift grid or the parent distribution.**

For example, a parent distribution defined over
:math:`0 \leq z \leq 3` may be divided into five bins covering only
:math:`0.2 \leq z \leq 1.2`.

The parent distribution remains defined over its original grid,
while the binning prescription determines the interval and placement
of the bin boundaries.


Binning modes
-------------

Binny supports three mutually exclusive ways of specifying bin edges:

- **Explicit edges:** the boundaries are supplied directly.
- **Generated edges:** a named scheme generates a requested number
  of bins.
- **Mixed binning:** multiple schemes are combined over adjacent
  redshift intervals.

All three modes use the same ``bins`` specification.

The central resolver is
:func:`binny.binning.edges.resolve_bin_edges`.


Explicit bin edges
~~~~~~~~~~~~~~~~~~

Explicit binning uses a supplied sequence of bin boundaries rather
than generating them from a mathematical prescription.

For :math:`N` bins, the specification must contain :math:`N+1`
strictly increasing edges,

.. math::

   z_0 < z_1 < \cdots < z_N.

This mode does not accept a generating scheme, a separate number
of bins, or an additional binning range.

See :ref:`example-explicit-edges` for a practical example.


Generated bin edges
~~~~~~~~~~~~~~~~~~~

Generated binning uses a named mathematical prescription to divide
a specified interval into :math:`N` bins.

The binning interval is

.. math::

   z_{\min} \leq z \leq z_{\max}.

If no interval is explicitly requested, the full range of the
input redshift grid is used.

A generated specification contains:

- ``scheme``: the name of the binning prescription.
- ``n_bins``: the requested number of bins.
- ``range``: an optional interval for constructing the edges.
- ``params``: optional parameters specific to the selected scheme.

Scheme-specific parameters belong inside ``params``, rather than
at the top level of the bin specification.

Unsupported scheme parameters raise a ``ValueError``.

See :ref:`example-generated-edges`.


Built-in binning schemes
------------------------

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


.. _theory-equidistant:

Equidistant binning
~~~~~~~~~~~~~~~~~~~

Equidistant binning divides the requested redshift interval into
equal-width bins.

For an interval from :math:`z_{\min}` to :math:`z_{\max}`,
the boundaries are

.. math::

   z_i =
   z_{\min}
   + \frac{i}{N}(z_{\max}-z_{\min}),
   \qquad i=0,\ldots,N.

Every bin has the same redshift width,

.. math::

   \Delta z =
   \frac{z_{\max}-z_{\min}}{N}.

The boundaries depend only on the requested interval and the
number of bins. They do not depend on the parent redshift
distribution.


.. _theory-equal-number:

Equal-number binning
~~~~~~~~~~~~~~~~~~~~

Equal-number binning selects boundaries such that each bin contains
the same integrated population.

For a parent distribution :math:`n(z)`, the boundaries satisfy

.. math::

   \int_{z_i}^{z_{i+1}} n(z)\,\mathrm{d}z
   =
   \frac{1}{N}
   \int_{z_{\min}}^{z_{\max}} n(z)\,\mathrm{d}z.

Unlike equidistant binning, the resulting bin widths generally
vary with redshift.

Regions with higher population density receive narrower bins,
while regions with lower population density receive wider bins.

The alias ``equipopulated`` is equivalent to ``equal_number``.

The resolver can also use a separate distribution for determining
equal-number boundaries through ``equal_number_axis`` and
``equal_number_weights``.

See :ref:`example-equal-number`.


.. _theory-generalized:

Generalized binning
-------------------

Generalized binning constructs boundaries by integrating a weight
function that depends on the parent redshift distribution and,
optionally, the comoving radial distance.

The generalized weight is

.. math::

   w(z) =
   [n(z)]^\alpha
   \left(
   \frac{\mathrm{d}\chi}{\mathrm{d}z}
   \right)^\beta,

where :math:`\chi(z)` is the comoving radial distance and
:math:`\alpha` and :math:`\beta` are configurable exponents.

The boundaries satisfy

.. math::

   \int_{z_i}^{z_{i+1}} w(z)\,\mathrm{d}z
   =
   \frac{1}{N}
   \int_{z_{\min}}^{z_{\max}} w(z)\,\mathrm{d}z.

This prescription contains several familiar binning strategies
as special cases.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - :math:`\alpha`
     - :math:`\beta`
     - Interpretation
   * - 0
     - 0
     - Equidistant redshift binning.
   * - 1
     - 0
     - Equal-number binning.
   * - 0
     - 1
     - Equal comoving-distance binning.

Intermediate exponent values allow bin boundaries to interpolate
between these prescriptions.

When :math:`\beta=0`, comoving-distance information is unnecessary.

For nonzero :math:`\beta`, the underlying
:func:`binny.axes.bin_edges.generalized_tomographic_edges`
function requires comoving-distance samples evaluated on the
same redshift grid.

The comoving-distance array must be strictly increasing and
compatible with the input coordinate grid.

When a restricted binning interval is specified, the resolver
interpolates the comoving-distance samples onto the restricted grid.

See :ref:`example-generalized`.


.. _theory-mixed-binning:

Mixed binning
-------------

Mixed binning combines different binning prescriptions across
adjacent redshift intervals.

Each segment specifies:

- A binning scheme.
- The number of bins in that segment.
- The lower and upper boundaries of the segment.

The total number of bins is the sum of the individual segment
counts,

.. math::

   N_{\mathrm{total}}
   =
   \sum_{s=1}^{N_{\mathrm{segments}}} N_s.

The resulting edges form one strictly increasing array.

Shared boundaries between adjacent segments are included only once.

Mixed binning uses the underlying
:func:`binny.axes.mixed_edges.mixed_edges` machinery.

For equal-number segments, the parent distribution is used as the
population proxy unless a separate selection distribution is
provided through ``selection_axis`` and ``selection_density``.

In photometric-redshift tomography, these may represent the
observed-redshift grid and its corresponding distribution.

See :ref:`example-mixed-binning`.


Custom binning schemes
----------------------

Binny supports user-defined binning prescriptions through a public
registry.

A custom scheme receives:

- The sampled coordinate axis.
- The corresponding density.
- The requested number of bins.
- Optional scheme-specific parameters.

It must return exactly :math:`N+1` strictly increasing bin edges
for :math:`N` requested bins.

Custom schemes use the same ``bins`` specification as built-in
schemes.

Scheme-specific arguments are supplied inside ``params`` and
validated against the registered function's signature.

Scheme names and aliases are case-insensitive.
Duplicate registered names are rejected.

The registry exposes:

- :func:`binny.binning.scheme_registry.register_binning_scheme`
- :func:`binny.binning.scheme_registry.get_binning_scheme`
- :func:`binny.binning.scheme_registry.available_binning_schemes`

See :ref:`example-custom-scheme` for registration and usage.


See also
--------

- :doc:`../examples/tomography/binning_schemes`
- :doc:`../examples/tomography/photoz_bins`
- :doc:`../examples/tomography/specz_bins`
- :doc:`tomography`
- :mod:`binny.binning.edges`
- :mod:`binny.axes.bin_edges`
