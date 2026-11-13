.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Mock redshift catalog sampling
=====================================

Binny provides utilities for generating mock redshift catalogs from
tabulated parent redshift distributions :math:`n(z)`.

The sampling machinery converts a continuous redshift distribution into
individual galaxy redshifts while preserving the statistical properties
of the input distribution. It supports arbitrary nonnegative tabulated
distributions, nonuniform redshift grids, reproducible random sampling,
and optional observational transformations.

The implementation uses inverse cumulative distribution function (CDF)
sampling with exact inversion of the piecewise-linear density between
grid nodes.

This provides a direct connection between theoretical redshift
distributions and object-level mock catalogs.

For executable examples, see the corresponding mock redshift sampling
examples page.


Overview
--------

Many cosmological calculations begin with a continuous parent redshift
distribution,

.. math::

   n(z),

which describes the relative abundance of galaxies as a function of
redshift.

Tomographic analyses generally operate on continuous distributions
or their integrals, whereas simulations, calibration studies, and
catalog-based workflows often require individual galaxy redshifts.

We denote the total number of sampled galaxies by :math:`N`, with
:math:`i=1,\ldots,N` indexing individual galaxies,

.. math::

   \{z_1,z_2,\ldots,z_N\}.

Binny connects these representations through the transformation

.. math::

   n(z) \longrightarrow \{z_i\}_{i=1}^{N}.

The resulting samples are independent draws from the probability
density defined by the supplied tabulated distribution.

The sampling process does not assume a particular survey, galaxy
population, or redshift measurement technique.


Probability density and normalization
-------------------------------------

The input distribution is specified at :math:`K` redshift grid nodes,
where :math:`K \geq 2` is the total number of nodes. The grid is
strictly increasing,

.. math::

   z_0 < z_1 < \cdots < z_{K-1}.

We use :math:`k=0,\ldots,K-1` to index the grid nodes, with
:math:`z_k` denoting the redshift at node :math:`k`.

The corresponding distribution values are nonnegative,

.. math::

   n_k = n(z_k) \geq 0.

The input distribution does not need to be normalized.

The corresponding probability density is

.. math::

   p(z) = \frac{n(z)}{\mathcal{N}},

where

.. math::

   \mathcal{N}
   = \int_{z_0}^{z_{K-1}} n(z)\,\mathrm{d}z.

The normalization must be positive and finite.

For numerical stability, Binny first rescales the input distribution
by its maximum value. Since the sampling probabilities depend only
on relative amplitudes, this rescaling does not change the resulting
probability distribution.

The normalization is then calculated using trapezoidal integration
over the supplied redshift grid.

The resulting density satisfies

.. math::

   \int_{z_0}^{z_{K-1}} p(z)\,\mathrm{d}z = 1,

up to floating-point precision.

Multiplying the input distribution by any positive constant therefore
does not change the sampling distribution.


Piecewise-linear representation
-------------------------------

Between adjacent grid nodes, Binny interprets the redshift
distribution as piecewise linear.

The index :math:`k=0,\ldots,K-2` labels the intervals between
consecutive nodes. There are therefore :math:`K-1` intervals
for :math:`K` grid nodes.

For an interval

.. math::

   z \in [z_k,z_{k+1}],

define its width as

.. math::

   \Delta z_k = z_{k+1}-z_k.

The local coordinate within the interval is

.. math::

   t = \frac{z-z_k}{\Delta z_k},
   \qquad 0 \leq t \leq 1.

The interpolated distribution is

.. math::

   n(z) = n_k + (n_{k+1}-n_k)t.

This convention defines the continuous probability distribution
from which the samples are drawn.

The probability mass associated with each interval is proportional
to its trapezoidal area,

.. math::

   A_k
   = \frac{n_k+n_{k+1}}{2}\Delta z_k.

The construction is valid for both uniform and nonuniform redshift
grids.

Importantly, the algorithm samples continuously within each
interval rather than restricting sampled redshifts to the original
grid nodes.


Cumulative distribution function
--------------------------------

The cumulative distribution function is

.. math::

   F(z) = \int_{z_0}^{z} p(z')\,\mathrm{d}z'.

At the grid nodes, the cumulative masses are constructed from the
trapezoidal interval contributions. Here :math:`j=0,\ldots,K-1`
indexes the node at which the cumulative mass is evaluated,
while :math:`k` indexes the intervals included in the sum.

.. math::

   C_0 = 0,

.. math::

   C_j = \sum_{k=0}^{j-1} A_k.

Here, :math:`C_j` is the cumulative unnormalized probability mass
up to redshift :math:`z_j`. The normalized CDF evaluated at that
node is denoted by :math:`F_j = F(z_j)`,

.. math::

   F_j = \frac{C_j}{C_{K-1}}.

Consequently,

.. math::

   F_0 = 0,
   \qquad
   F_{K-1} = 1.

Binny uses its existing ``cdf_from_curve`` utility to construct
the cumulative trapezoidal masses and their normalization.

This ensures consistency with the numerical conventions used
elsewhere in the package.


Inverse-CDF sampling
--------------------

Inverse-CDF sampling generates each redshift by first drawing a
random number :math:`u` from the continuous uniform distribution
:math:`\mathcal{U}(0,1)`,

.. math::

   u \sim \mathcal{U}(0,1)

and solving

.. math::

   F(z) = u.

For each random variate, Binny first identifies the interval
containing the corresponding cumulative probability.

If

.. math::

   F_k \leq u < F_{k+1},

the sample lies in the interval

.. math::

   z \in [z_k,z_{k+1}].

The normalized position in cumulative probability within that
interval is

.. math::

   q = \frac{u-F_k}{F_{k+1}-F_k}.

A direct linear interpolation of the CDF would generally be
incorrect because the underlying probability density is
piecewise linear rather than piecewise constant.

Instead, Binny analytically inverts the integrated
piecewise-linear density within each interval.


Exact inversion within an interval
----------------------------------

Let the endpoint densities of the selected interval be

.. math::

   a = n_k,
   \qquad
   b = n_{k+1}.

Using the local coordinate :math:`t`, the density is

.. math::

   n(t) = a+(b-a)t.

The cumulative mass from the left endpoint to :math:`t` is
proportional to

.. math::

   \int_0^t [a+(b-a)s]\,\mathrm{d}s
   = at+\frac{b-a}{2}t^2.

The total mass of the interval is proportional to

.. math::

   \frac{a+b}{2}.

The local cumulative probability therefore satisfies

.. math::

   q =
   \frac{at+\frac{1}{2}(b-a)t^2}
        {\frac{1}{2}(a+b)}.

Rearranging gives the quadratic equation

.. math::

   (b-a)t^2+2at-q(a+b)=0.

Define

.. math::

   T = \frac{q(a+b)}{2}.

The physically relevant solution can be written in the
numerically stable form

.. math::

   t =
   \frac{2T}
        {a+\sqrt{a^2+2(b-a)T}}.

The sampled redshift is then

.. math::

   z = z_k+t\Delta z_k.

This inversion is exact for the piecewise-linear distribution
represented by the input grid, up to floating-point precision.

For constant-density intervals, where :math:`a=b`, the
solution simplifies to

.. math::

   t=q.

Zero-probability intervals do not contribute to the sampling
distribution.


Statistical interpretation
--------------------------

The sampled catalog represents a Monte Carlo realization of
the continuous probability distribution.

For a sufficiently large number of samples, the empirical
redshift distribution approaches the normalized input
distribution.

Consider an arbitrary redshift interval :math:`B` within the
sampling range. Let :math:`P(z\in B)` denote the probability
that a sampled galaxy lies inside this interval. This probability
is given by

.. math::

   P(z\in B) = \int_B p(z)\,\mathrm{d}z.

Let :math:`N_B` denote the random number of galaxies among the
:math:`N` samples that fall inside :math:`B`. Its expectation
:math:`\mathbb{E}[N_B]` is

.. math::

   \mathbb{E}[N_B] = N P(z\in B).

For independent samples, the corresponding variance is

.. math::

   \operatorname{Var}(N_B)
   = NP(z\in B)[1-P(z\in B)].

Consequently, a finite sampled catalog will not reproduce
the input distribution exactly.

Differences between the empirical histogram and the
underlying probability density are expected Monte Carlo
fluctuations rather than errors in the sampling procedure.

The quality of the representation also depends on the
resolution of the original redshift grid. The algorithm
samples exactly from the piecewise-linear interpolation,
not from an unknown continuous function between grid nodes.


Random-number generation
------------------------

Binny supports reproducible sampling through either a
random seed or an existing NumPy random generator.

The two options are mutually exclusive.

When a seed is supplied, Binny creates a NumPy
``Generator`` using ``numpy.random.default_rng``.

When an existing generator is supplied, its state is
advanced by the sampling operation.

Using the same seed, input distribution, and sample count
reproduces the same redshift realization under the same
random-number generation implementation.

Passing an existing generator is particularly useful when
redshift sampling is part of a larger simulation workflow
that requires a shared random-number stream.


Mock catalog construction
-------------------------

In addition to returning an array of sampled redshifts,
Binny provides a higher-level mock catalog utility.

The catalog is represented as a dictionary of NumPy arrays.

The default output contains:

``z``
   Sampled redshifts drawn from the input distribution.

``weight``
   Per-object weights, equal to one by default.

The catalog helper does not impose a particular physical
interpretation on the input distribution.

For example, when the supplied :math:`n(z)` describes the
underlying true-redshift population, the sampled ``z``
values represent true redshifts.

If the input distribution instead describes another
redshift population, the sampled values inherit that
interpretation.

This distinction is important when constructing mock
observations.


Object weights
--------------

Each sampled object can optionally carry a weight,

.. math::

   w_i \geq 0.

By default,

.. math::

   w_i = 1.

Users can provide either a single scalar weight applied
to every sampled object or a one-dimensional array
containing one weight per object.

The weights must be finite and nonnegative.

Weights are attached after the redshift sampling step.
They do not modify the probability distribution used
to generate the sampled redshifts.

A weighted empirical redshift distribution can be represented
by :math:`\hat{p}_w(z)`, where the hat denotes an empirical
estimate and the subscript :math:`w` indicates weighting.
Using the Dirac delta distribution :math:`\delta_{\rm D}`,
which represents a point contribution at each sampled
redshift, we write

.. math::

   \hat{p}_w(z)
   = \frac{1}{\sum_i w_i}
     \sum_{i=1}^{N} w_i\,\delta_{\rm D}(z-z_i),

provided the total weight is positive.

This is useful for representing selection weights,
statistical weights, or survey-specific catalog
weighting schemes.


Observation models
------------------

The mock catalog utility optionally accepts an observation
model that transforms sampled redshifts into observed
redshifts.

Conceptually, an observation model describes a conditional
distribution,

.. math::

   p(z_{\rm obs}\mid z).

For an underlying true-redshift distribution, the
corresponding observed-redshift distribution is

.. math::

   p_{\rm obs}(z_{\rm obs})
   = \int p(z_{\rm obs}\mid z)
          p(z)\,\mathrm{d}z.

At the catalog level, this corresponds to drawing

.. math::

   z_i \sim p(z),

followed by

.. math::

   z_{{\rm obs},i}
   \sim p(z_{\rm obs}\mid z_i).

The observation model is supplied as a callable that
accepts the sampled redshift array and a NumPy random
generator.

The callable must return one finite observed redshift
per sampled object.

When provided, the catalog contains the additional
column ``z_obs``.

The same random generator is used for both the
underlying redshift samples and the observation model.

This makes the entire mock catalog generation procedure
reproducible with a single seed.


Redshift measurement models
---------------------------

The observation-model interface is intentionally general.

It can represent different redshift measurement techniques,
including:

- photometric redshift estimates;
- spectroscopic redshift measurements;
- grism or narrow-band redshift estimates;
- user-defined redshift measurement models.

For example, a Gaussian measurement model may be written as

.. math::

   z_{\rm obs}
   = z+\mu(z)+\sigma(z)\epsilon,

where

.. math::

   \epsilon\sim\mathcal{N}(0,1).

Here, :math:`\mu(z)` describes a redshift-dependent
systematic offset and :math:`\sigma(z)` describes the
measurement scatter.

More general observation models can include non-Gaussian
tails, asymmetric errors, catastrophic outliers, or
population-dependent measurement uncertainties.

The catalog sampler itself does not prescribe these
models.

Instead, the observation-model callable defines the
measurement transformation.

This avoids introducing a second, independent photo-z
error prescription into the sampling utilities.


Connection to photo-z tomography
--------------------------------

Binny already provides photo-z error modelling and
tomographic selection machinery.

The catalog sampling utilities complement this
infrastructure by generating individual redshift
realizations from a continuous parent distribution.

The two operations are conceptually distinct:

- parent distribution sampling generates the underlying
  object-level redshift population;
- an observation model generates measured redshifts
  conditional on those sampled values;
- tomographic selection assigns objects or probabilities
  to redshift bins.


For a tomographic bin :math:`i`, where :math:`i` now labels
tomographic bins rather than individual galaxies, let
:math:`z_{{\rm obs},i}^{\rm min}` and
:math:`z_{{\rm obs},i}^{\rm max}` denote its lower and upper
observed-redshift boundaries, respectively.

The probability :math:`P_i(z)` that a galaxy at true redshift
:math:`z` is assigned to bin :math:`i` is

.. math::

   P_i(z)
   = \int_{z_{{\rm obs},i}^{\rm min}}^
          {z_{{\rm obs},i}^{\rm max}}
     p(z_{\rm obs}\mid z)\,\mathrm{d}z_{\rm obs}.

The expected true-redshift distribution of objects
assigned to that bin is then proportional to

.. math::

   n_i(z) = n(z)P_i(z).

At the catalog level, the analogous procedure is
to sample underlying redshifts, generate observed
redshifts, and apply the corresponding selection
criteria.

Consistency between catalog-based and analytic
tomography requires using the same conditional
redshift-error model and selection conventions.

The generic observation-model interface makes such
workflows possible without embedding a separate
photo-z model directly in the catalog sampler.

The current catalog utility does not automatically
generate tomographic bin assignments or outlier
classification flags.


Input validation and numerical considerations
---------------------------------------------

The redshift sampler validates that the supplied
redshift grid is one-dimensional, finite, strictly
increasing, and contains at least two nodes.

Redshift values must be nonnegative.

The tabulated distribution must have the same shape
as the redshift grid and contain finite, nonnegative
values with a positive integral.

The number of requested samples must be a positive
integer.

The returned redshifts are continuous floating-point
values within the supplied redshift range.

The implementation handles nonuniform grids and
piecewise-linear distributions with zero-density
regions.

As with any floating-point numerical procedure,
extreme dynamic ranges and very small cumulative
probability intervals can be sensitive to finite
precision.

For typical survey redshift distributions, the
method provides an efficient vectorized approach
to generating large mock catalogs.


Relationship to analytic parent models
--------------------------------------

The sampler operates on tabulated distributions and
does not require a particular analytic model.

An analytic parent model can first be evaluated on
a redshift grid. Denoting its model parameters collectively
by the vector :math:`\boldsymbol{\theta}`, we write

.. math::

   n(z_k;\boldsymbol{\theta}),

and the resulting tabulated curve can then be
passed to the sampler.

This supports mock catalog generation from the
analytic parent models provided by Binny, including
Smail-like, Gaussian, mixture, and other supported
redshift distributions.

The same procedure applies to parent distributions
calibrated from mock catalogs or constructed from
luminosity-function-weighted models.

Thus, the sampling utilities are independent of
the physical or phenomenological origin of the
input :math:`n(z)`.


Summary
-------

Binny's mock redshift sampling utilities connect
continuous parent redshift distributions to
object-level mock catalogs.

The implementation provides:

- exact inverse-CDF sampling for piecewise-linear
  tabulated redshift distributions;
- automatic normalization of nonnegative inputs;
- support for nonuniform redshift grids;
- reproducible sampling with seeds or NumPy generators;
- optional per-object weights;
- optional observed-redshift generation through
  user-defined observation models;
- dictionary-based mock catalog outputs.

The approach keeps parent distribution modelling,
redshift measurement modelling, and tomographic
selection conceptually separate.

For executable usage examples, see the mock
redshift sampling examples page.
