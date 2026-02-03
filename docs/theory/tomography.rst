Tomographic binning
===================

Tomography in cosmological analyses refers to the practice of subdividing a
galaxy sample into redshift bins and measuring correlations both within and
between these bins. By retaining partial redshift information, tomographic
analyses recover information that would otherwise be lost in fully projected
(two-dimensional) measurements.

At the core of tomography is the redshift distribution of galaxies,
:math:`n(z)`, which describes the number density of objects as a function of
redshift. A tomographic binning scheme partitions this distribution into a set
of bins indexed by :math:`i = 1, \ldots, N_{\mathrm{bin}}`.

Each bin is defined by a window function :math:`W_i(z)`, such that the binned
distribution is

.. math::

   n_i(z) = n(z)\, W_i(z).

In the simplest case, the window function corresponds to a hard redshift cut,

.. math::

   W_i(z) =
   \begin{cases}
     1 & z_i^{\mathrm{min}} \le z < z_i^{\mathrm{max}}, \\
     0 & \text{otherwise},
   \end{cases}

though more general (e.g. probabilistic or overlapping) assignments are commonly
used in photometric surveys.

---

Motivation for tomography
-------------------------

Tomographic binning allows cosmological correlations to be studied as a function
of redshift, thereby probing the time evolution of large-scale structure. This
is particularly important for observables that integrate information along the
line of sight, such as galaxy clustering or weak gravitational lensing.

By splitting the galaxy sample into multiple bins, one gains access to:

- **Auto-correlations**, measured between galaxies within the same redshift bin;
- **Cross-correlations**, measured between galaxies in different redshift bins.

The joint analysis of auto- and cross-correlations enables sensitivity to
redshift-dependent physical effects, such as the growth of structure, geometric
distances, and bias evolution, that are partially degenerate in fully projected
measurements.

The general rationale for tomographic analyses in cosmology was laid out in the
context of weak gravitational lensing by Hu (1999), who showed that even coarse
redshift binning can recover a large fraction of the available three-dimensional
information :contentReference[oaicite:1]{index=1}.

---

Spectroscopic versus photometric tomography
--------------------------------------------

Both spectroscopic and photometric surveys employ tomographic techniques, but
with important differences.

In **spectroscopic surveys**, redshifts are measured with high precision, making
it possible to define bins with negligible overlap in true redshift. In this
case, the window functions :math:`W_i(z)` can be treated as sharply bounded.

In **photometric surveys**, redshifts are estimated with finite uncertainty.
As a result, the effective redshift distributions :math:`n_i(z)` of different
bins generally overlap, even when the bin definitions themselves do not.
Tomographic binning in this context therefore involves a trade-off between
redshift resolution and statistical noise.

Despite these limitations, photometric tomography has been shown to retain much
of the constraining power of full three-dimensional analyses, provided the bins
are chosen appropriately and the overlap between bins is accounted for in the
analysis.

---

Choice of binning scheme
------------------------

There is no unique optimal tomographic binning scheme. Common choices include:

- Equally spaced bins in redshift;
- Equal-number (equipopulated) bins;
- Hybrid or segmented schemes combining multiple criteria.

The optimal choice depends on the redshift distribution of the sample, the
observable being studied, and the scientific goals of the analysis. In practice,
relatively coarse binning is often sufficient to extract most of the available
information, especially when correlations between bins are strong.

Binny is designed to make these binning choices explicit and reproducible, and
to provide tools for constructing, validating, and comparing different
tomographic schemes in a consistent framework.
