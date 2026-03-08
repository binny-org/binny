.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Tomography
=================

Tomography in cosmological analyses refers to the practice of subdividing a
galaxy sample into redshift bins and measuring correlations both within and
between these bins. By retaining partial redshift information, tomographic
analyses recover information that would otherwise be lost in fully projected
(two-dimensional) measurements.

At the core of tomography is the redshift distribution of galaxies,
:math:`n(z)`, which describes the number density of objects (galaxies) as a
function of redshift :math:`z`. A tomographic binning scheme partitions this
distribution into a set of bins indexed by
:math:`i = 1, \ldots, N_{\mathrm{bin}}`.

Each bin is defined through a window function :math:`W_i(z)`, such that the
binned distribution becomes

.. math::

   n_i(z) = n(z)\, W_i(z).

Here

- :math:`z` denotes the redshift,
- :math:`n(z)` is the **parent redshift distribution**, describing the number
  density of galaxies as a function of redshift before binning,
- :math:`W_i(z)` is the **window function** for bin :math:`i`, specifying how
  galaxies at redshift :math:`z` contribute to that bin,
- :math:`n_i(z)` is the resulting **binned redshift distribution** for bin
  :math:`i`,
- :math:`N_{\mathrm{bin}}` is the total number of tomographic bins in the
  analysis.

The window functions determine which galaxies contribute to each bin and how
their contributions are weighted.

In the simplest case, the window function corresponds to a hard redshift cut,

.. math::

   W_i(z) =
   \begin{cases}
     1 & z_i^{\mathrm{min}} \le z < z_i^{\mathrm{max}}, \\
     0 & \text{otherwise}.
   \end{cases}

where :math:`z_i^{\mathrm{min}}` and :math:`z_i^{\mathrm{max}}` denote the lower
and upper redshift boundaries of bin :math:`i`.

More general window functions are often used in practice, for example when bins
overlap or when galaxies contribute probabilistically to multiple bins.


Motivation for tomography
-------------------------

Tomographic binning allows cosmological correlations to be studied as a function
of redshift, thereby probing the time evolution of large-scale structure. This
is particularly important for observables that integrate information along the
line of sight, such as galaxy clustering or weak gravitational lensing.

By splitting the galaxy sample into multiple bins, one gains access to

- **auto-correlations**, measured between galaxies within the same redshift bin;
- **cross-correlations**, measured between galaxies in different redshift bins.

The joint analysis of auto- and cross-correlations enables sensitivity to
redshift-dependent physical effects such as the growth of structure, geometric
distances, and galaxy bias evolution. These effects are partially degenerate in
fully projected measurements but become separable when redshift information is
retained.

Tomographic weak-lensing analyses were formalized by Hu (1999) [Hu1999]_,
who showed that even coarse redshift binning can recover a large
fraction of the available three-dimensional information.


Spectroscopic vs photometric tomography
---------------------------------------

Tomographic analyses are used in both spectroscopic and photometric surveys,
though the practical implementation differs.

Spectroscopic tomography
^^^^^^^^^^^^^^^^^^^^^^^^

In spectroscopic surveys, redshifts are measured with high precision.
Tomographic bins can therefore be defined with minimal overlap in true redshift,
and the corresponding window functions are often treated as sharply bounded.

Photometric tomography
^^^^^^^^^^^^^^^^^^^^^^

In photometric surveys, redshifts are estimated from galaxy colors rather than
spectral lines. The resulting bins are typically broader in true redshift and
may overlap even when the nominal bin edges are well separated.

Despite these differences, the underlying tomographic framework is the same:
the galaxy population is partitioned into bins and correlations are measured
within and across these bins.


Binning schemes
---------------

A tomographic analysis requires a rule for defining the bin boundaries
:math:`\{z_i^{\mathrm{min}}, z_i^{\mathrm{max}}\}`. Several binning strategies
are commonly used.

Equidistant binning
^^^^^^^^^^^^^^^^^^^

In equidistant binning, the redshift interval is divided into bins of equal
width,

.. math::

   z_i^{\mathrm{min}} = z_{\mathrm{min}} + (i-1)\Delta z,
   \qquad
   z_i^{\mathrm{max}} = z_{\mathrm{min}} + i\,\Delta z,

where :math:`\Delta z = (z_{\mathrm{max}} - z_{\mathrm{min}})/N_{\mathrm{bin}}`.

This scheme provides uniform redshift coverage and is frequently used when the
analysis requires a simple geometric partition of the redshift range.

Equipopulated binning
^^^^^^^^^^^^^^^^^^^^^

In equipopulated binning, the bin edges are chosen such that each bin contains
approximately the same fraction of galaxies,

.. math::

   \int_{z_i^{\mathrm{min}}}^{z_i^{\mathrm{max}}} n(z)\,\mathrm{d}z
   \approx
   \frac{1}{N_{\mathrm{bin}}}
   \int n(z)\,\mathrm{d}z.

This approach produces bins with comparable statistical weight and is commonly
used in photometric weak-lensing analyses.

Segmented or mixed binning
^^^^^^^^^^^^^^^^^^^^^^^^^^

More flexible schemes can be constructed by combining different binning
strategies across redshift segments. For example, one may apply equal-number
binning at low redshift while switching to equidistant bins at higher redshift.

Such hybrid approaches allow the binning scheme to adapt to features in the
underlying redshift distribution while preserving control over bin boundaries.


Role of binning in cosmological analyses
----------------------------------------

The tomographic bins define the set of observables used in a cosmological
analysis. For example, a tomographic clustering measurement produces angular
power spectra

.. math::

   C_\ell^{ij}

for all combinations of bins :math:`i` and :math:`j`.

Similarly, weak-lensing tomography produces shear correlations between source
bins, while joint analyses may combine clustering, galaxy–galaxy lensing, and
cosmic shear measurements across multiple bin pairs.

The number of bins and the choice of binning scheme therefore determine both
the dimensionality of the data vector and the redshift resolution of the
resulting cosmological constraints.


Binning in Binny
----------------

Binny provides tools for constructing tomographic bins directly from a parent
redshift distribution :math:`n(z)`. Several binning schemes are supported,
including

- **equidistant bins** (uniform redshift spacing),
- **equal-number bins** (approximately equal galaxy counts per bin),
- **mixed segmented schemes** that combine multiple strategies.

The resulting bins are represented as curves :math:`n_i(z)` on a shared redshift
grid, allowing them to be analyzed, visualized, and compared consistently across
different binning configurations.


References
----------
.. [Hu1999] Hu, W. (1999),
   *Power Spectrum Tomography with Weak Lensing*,
   ApJL 522, L21–L24.
   https://arxiv.org/abs/astro-ph/9904153