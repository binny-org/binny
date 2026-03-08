.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Redshift uncertainty models
==================================

This page describes how Binny implements redshift-selection models for
constructing tomographic bins from a parent redshift distribution
:math:`n(z)`.

For a conceptual overview of tomography, binning schemes, and the role of
tomographic observables in cosmology, see :doc:`tomography`.

Binny supports two broad classes of redshift-selection model:

- **Photo-z tomography**, where bins are defined in observed redshift and
  mapped back onto the true-redshift grid through a probabilistic assignment
  model.
- **Spec-z tomography**, where bins are defined in true redshift, with optional
  completeness losses and bin-to-bin response effects that model residual
  spectroscopic failures or measurement scatter.

In both cases, the returned tomographic bins are evaluated on a common
true-redshift grid :math:`z`.


Overview
--------

Binny constructs tomographic bins by applying an effective redshift-selection
model to a parent redshift distribution :math:`n(z)`. The resulting bin curves
may be written schematically as

.. math::

   n_i(z) = n(z)\, S_i(z),

where

- :math:`z` is the **true redshift**,
- :math:`n(z)` is the **parent redshift distribution** evaluated on the
  true-redshift grid,
- :math:`i` is the **tomographic-bin index**,
- :math:`n_i(z)` is the **bin distribution** associated with bin :math:`i`,
- :math:`S_i(z)` is the **effective selection function** for bin :math:`i`.

The interpretation of :math:`S_i(z)` depends on the redshift model:

- in the **photo-z** case, :math:`S_i(z)` is the bin-assignment probability
  :math:`P(i \mid z)`,
- in the **spec-z** case, :math:`S_i(z)` is typically a top-hat-like
  true-redshift selection, optionally followed by a response model.

Conceptually, the key distinction is:

- **photo-z tomography** is probabilistic from the outset, because bins are
  defined in observed redshift;
- **spec-z tomography** begins as a deterministic partition in true redshift,
  with optional response effects layered on top.

If requested, Binny also records metadata describing pre-normalization bin
weights, so the user can distinguish between bin *shape* and bin *population*.


Photometric-redshift model
--------------------------

In photometric-redshift tomography, tomographic bins are defined in observed
redshift :math:`z_{\mathrm{ph}}`, but the returned bin curves are evaluated on
the true-redshift grid :math:`z`.

For an observed-redshift bin with edges
:math:`[z_{\mathrm{ph,min},i}, z_{\mathrm{ph,max},i}]`, the selected
true-redshift distribution is

.. math::

   n_i(z) = n(z)\, P(i \mid z),

where

- :math:`n_i(z)` is the returned tomographic bin on the **true-redshift** grid,
- :math:`n(z)` is the parent redshift distribution,
- :math:`P(i \mid z)` is the probability that an object at true redshift
  :math:`z` is assigned to observed bin :math:`i`,
- :math:`z_{\mathrm{ph,min},i}` and :math:`z_{\mathrm{ph,max},i}` are the lower
  and upper observed-redshift edges of bin :math:`i`.

This means that photo-z tomography does **not** apply a hard cut in true
redshift. Instead, each true-redshift value contributes to a bin according to
the photo-z assignment model.


Core Gaussian assignment
~~~~~~~~~~~~~~~~~~~~~~~~

The core photo-z model assumes that the observed redshift
:math:`z_{\mathrm{ph}}` at fixed true redshift :math:`z` follows a Gaussian
distribution:

.. math::

   z_{\mathrm{ph}} \sim \mathcal{N}\!\bigl(\mu(z), \sigma(z)\bigr),

where

- :math:`z_{\mathrm{ph}}` is the **observed** or photometric redshift,
- :math:`z` is the **true** redshift,
- :math:`\mu(z)` is the mean observed-redshift relation at fixed true redshift,
- :math:`\sigma(z)` is the scatter of the observed-redshift relation,
- :math:`\mathcal{N}(\mu,\sigma)` denotes a Gaussian distribution with mean
  :math:`\mu` and standard deviation :math:`\sigma`.

In Binny, the mean relation is defined as a simple linear function of true redshift:

.. math::

   \mu(z) = \alpha z - \beta,

where

- :math:`\alpha` is the multiplicative mean-scaling parameter,
- :math:`\beta` is the additive mean-offset parameter.

In the API, these correspond to:

- :math:`\alpha`  : ``mean_scale``
- :math:`\beta`   : ``mean_offset``

The scatter model is

.. math::

   \sigma(z) = s\,(1+z),

where

- :math:`\sigma(z)` is the Gaussian standard deviation in observed redshift,
- :math:`s` is the scatter amplitude.

In the API:

- :math:`s` : ``scatter_scale``

The corresponding conditional density is

.. math::

   p(z_{\mathrm{ph}} \mid z)
   =
   \frac{1}{\sqrt{2\pi}\,\sigma(z)}
   \exp\!\left[
      -\frac{\bigl(z_{\mathrm{ph}}-\mu(z)\bigr)^2}{2\sigma^2(z)}
   \right],

where

- :math:`p(z_{\mathrm{ph}} \mid z)` is the conditional probability density of
  observed redshift at fixed true redshift.

The bin-assignment probability is obtained by integrating this density between
the observed-redshift edges of bin :math:`i`:

.. math::

   P(i \mid z)
   =
   \int_{z_{\mathrm{ph,min},i}}^{z_{\mathrm{ph,max},i}}
   p(z_{\mathrm{ph}} \mid z)\, \mathrm{d}z_{\mathrm{ph}},

where

- :math:`P(i \mid z)` is the probability of assigning a galaxy at true redshift
  :math:`z` to observed bin :math:`i`,
- :math:`\mathrm{d}z_{\mathrm{ph}}` is the integration measure in observed
  redshift.

Because the model is Gaussian, this integral can be written analytically as

.. math::

   P(i \mid z)
   =
   \frac{1}{2}
   \left[
      \operatorname{erf}\!\left(
         \frac{z_{\mathrm{ph,max},i}-\mu(z)}{\sqrt{2}\,\sigma(z)}
      \right)
      -
      \operatorname{erf}\!\left(
         \frac{z_{\mathrm{ph,min},i}-\mu(z)}{\sqrt{2}\,\sigma(z)}
      \right)
   \right],

where

- :math:`\operatorname{erf}` is the error function.

In the implementation, this analytic form is used directly, making the
assignment smooth, fast, and numerically stable.


No-uncertainty limit
~~~~~~~~~~~~~~~~~~~~

If the scatter amplitude vanishes, :math:`s = 0`, the Gaussian model reduces
to a deterministic mapping. In that limit,

.. math::

   z_{\mathrm{ph}} = \mu(z) = \alpha z - \beta,

where

- :math:`z_{\mathrm{ph}}` is now a deterministic function of true redshift,
- :math:`\alpha` and :math:`\beta` retain the meanings defined above.

The assignment probability becomes

.. math::

   P(i \mid z) =
   \begin{cases}
   1, & z_{\mathrm{ph}} \in [z_{\mathrm{ph,min},i}, z_{\mathrm{ph,max},i}) \\
   0, & \text{otherwise.}
   \end{cases}

Here

- :math:`1` indicates certain assignment to bin :math:`i`,
- :math:`0` indicates that the object is not assigned to bin :math:`i`.

This is the sharp-selection limit of the photo-z construction.


Photo-z uncertainty terms
~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented photo-z model includes several distinct uncertainty terms.


Scatter
^^^^^^^

The scatter model is

.. math::

   \sigma(z) = s\,(1+z),

where

- :math:`\sigma(z)` is the Gaussian width in observed redshift,
- :math:`s` is the scatter amplitude.

API mapping:

- :math:`s` : ``scatter_scale``

Larger :math:`s` values broaden the assignment probability
:math:`P(i \mid z)`, increase overlap between neighboring tomographic bins,
and produce less sharply localized true-redshift distributions.


Bias / mean shift
^^^^^^^^^^^^^^^^^

The mean relation includes an additive offset,

.. math::

   \mu(z) = \alpha z - \beta,

where

- :math:`\beta` is the additive shift in the mean photo-z relation.

API mapping:

- :math:`\beta` : ``mean_offset``

Changing :math:`\beta` shifts the mapping between true redshift and observed
redshift and therefore shifts the location of the tomographic selection.


Mean scaling
^^^^^^^^^^^^

The same mean relation also includes a multiplicative scaling,

.. math::

   \mu(z) = \alpha z - \beta,

where

- :math:`\alpha` is the multiplicative scaling of true redshift in the mean
  relation.

API mapping:

- :math:`\alpha` : ``mean_scale``

When :math:`\alpha \neq 1`, the mapping between true and observed redshift is
stretched or compressed with redshift.


Outlier mixture
^^^^^^^^^^^^^^^

Binny also supports a second Gaussian component representing catastrophic or
outlier-like photo-z failures. The full conditional density is then

.. math::

   p(z_{\mathrm{ph}} \mid z)
   =
   (1-f_{\mathrm{out}})
   p_{\mathrm{core}}(z_{\mathrm{ph}} \mid z)
   +
   f_{\mathrm{out}}
   p_{\mathrm{out}}(z_{\mathrm{ph}} \mid z),

where

- :math:`p_{\mathrm{core}}(z_{\mathrm{ph}} \mid z)` is the core Gaussian
  conditional density,
- :math:`p_{\mathrm{out}}(z_{\mathrm{ph}} \mid z)` is the outlier Gaussian
  conditional density,
- :math:`f_{\mathrm{out}}` is the outlier fraction,
- :math:`1-f_{\mathrm{out}}` is the fraction assigned to the core component.

API mapping:

- :math:`f_{\mathrm{out}}` : ``outlier_frac``

Equivalently, the assignment probability becomes

.. math::

   P(i \mid z)
   =
   (1-f_{\mathrm{out}})\, P_{\mathrm{core}}(i \mid z)
   +
   f_{\mathrm{out}}\, P_{\mathrm{out}}(i \mid z),

where

- :math:`P_{\mathrm{core}}(i \mid z)` is the core assignment probability,
- :math:`P_{\mathrm{out}}(i \mid z)` is the outlier assignment probability.

The outlier component has its own mean and scatter model:

.. math::

   \mu_{\mathrm{out}}(z) = \alpha_{\mathrm{out}} z - \beta_{\mathrm{out}},

.. math::

   \sigma_{\mathrm{out}}(z) = s_{\mathrm{out}} (1+z),

where

- :math:`\mu_{\mathrm{out}}(z)` is the outlier mean relation,
- :math:`\sigma_{\mathrm{out}}(z)` is the outlier scatter,
- :math:`\alpha_{\mathrm{out}}` is the outlier mean scaling,
- :math:`\beta_{\mathrm{out}}` is the outlier mean offset,
- :math:`s_{\mathrm{out}}` is the outlier scatter amplitude.

API mapping:

- :math:`\alpha_{\mathrm{out}}` : ``outlier_mean_scale``
- :math:`\beta_{\mathrm{out}}`  : ``outlier_mean_offset``
- :math:`s_{\mathrm{out}}`      : ``outlier_scatter_scale``

This component allows a minority population to follow a broader, shifted, or
otherwise distorted redshift-assignment model relative to the core sample.


Implemented photo-z parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following photo-z parameters are implemented in Binny:

- :math:`s` : ``scatter_scale`` — core Gaussian scatter amplitude
- :math:`\beta` : ``mean_offset`` — additive shift in the core mean relation
- :math:`\alpha` : ``mean_scale`` — multiplicative scaling in the core mean relation
- :math:`f_{\mathrm{out}}` : ``outlier_frac`` — outlier-component fraction
- :math:`s_{\mathrm{out}}` : ``outlier_scatter_scale`` — outlier Gaussian scatter amplitude
- :math:`\beta_{\mathrm{out}}` : ``outlier_mean_offset`` — additive shift in the outlier mean relation
- :math:`\alpha_{\mathrm{out}}` : ``outlier_mean_scale`` — multiplicative scaling in the outlier mean relation

Each of these may be provided either as a scalar, applying the same value to
all bins, or as a per-bin sequence.


Interpreting the returned photo-z bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The returned photo-z tomographic bins are distributions on the **true**
redshift grid. They should therefore be interpreted as

.. math::

   n_i(z) = n(z)\, P(i \mid z),

where

- :math:`n_i(z)` is the returned bin distribution,
- :math:`n(z)` is the parent distribution,
- :math:`P(i \mid z)` is the true-to-observed bin-assignment probability.

They are therefore **not** histograms directly in observed-redshift space.

If ``normalize_bins=True``, each bin is normalized to integrate to unity and
becomes a shape-only redshift probability density. In that case, relative bin
populations are no longer encoded in the returned curves and should instead be
read from the metadata.


Metadata and population fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If metadata is requested, Binny records how much of the original parent
distribution falls into each observed bin **before** any per-bin
normalization.

The stored quantities include:

- ``parent_norm``: total area under the parent distribution :math:`n(z)`
- ``bins_norms[i]``: area under the raw bin curve for bin :math:`i` before normalization
- ``frac_per_bin[i]``: fractional population in bin :math:`i`, computed as
  ``bins_norms[i] / parent_norm`` when the parent norm is nonzero

This distinction is important: once individual bins are normalized, they retain
their redshift *shape* but no longer retain their relative *abundance*.


Spectroscopic-redshift model
----------------------------

In spectroscopic-redshift tomography, bins are defined directly in true
redshift. In the simplest case, a tomographic bin is a top-hat selection over a
true-redshift interval.

For true-redshift edges :math:`[z_j, z_{j+1}]`, the true-bin selection window is

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),

where

- :math:`S_j(z)` is the true-redshift selection function for true bin :math:`j`,
- :math:`c_j` is the completeness factor for bin :math:`j`,
- :math:`\mathbf{1}_{[z_j, z_{j+1})}(z)` is the indicator function, equal to
  :math:`1` when :math:`z \in [z_j, z_{j+1})` and :math:`0` otherwise,
- :math:`z_j` and :math:`z_{j+1}` are the lower and upper edges of true bin
  :math:`j`.

API mapping:

- :math:`c_j` : ``completeness``

The corresponding true-bin distribution is

.. math::

   n_j^{\mathrm{true}}(z) = n(z)\, S_j(z),

where

- :math:`n_j^{\mathrm{true}}(z)` is the true-bin distribution for bin
  :math:`j`,
- :math:`n(z)` is the parent distribution.

In the absence of any response effects, the observed and true bins coincide:

.. math::

   n_i^{\mathrm{obs}}(z) = n_i^{\mathrm{true}}(z),

where

- :math:`n_i^{\mathrm{obs}}(z)` is the observed-bin distribution,
- :math:`n_i^{\mathrm{true}}(z)` is the true-bin distribution.

This is the deterministic idealization of spectroscopic tomography.


Completeness uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~

The simplest spec-z uncertainty ingredient is the completeness factor:

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),
   \qquad 0 \le c_j \le 1,

where

- :math:`c_j=1` corresponds to full completeness in bin :math:`j`,
- :math:`c_j<1` reduces the pre-normalization population in that bin.

API mapping:

- :math:`c_j` : ``completeness``

Because completeness acts multiplicatively, it changes bin populations before
optional normalization. If bins are later normalized individually, the shape of
a nonempty bin is preserved, while the population loss remains visible only
through metadata.


Observed-bin response
~~~~~~~~~~~~~~~~~~~~~

Although spec-z bins are defined in true redshift, Binny also supports response
effects that map true bins into observed bins. This is useful for modeling bin
reassignment, imperfect classification, or leakage between bins.

The observed-bin distributions are constructed from the true-bin distributions
through a response matrix :math:`M`:

.. math::

   M_{ij} = P(i_{\mathrm{obs}} \mid j_{\mathrm{true}}),

where

- :math:`M_{ij}` is the probability that an object in true bin :math:`j`
  appaers in observed bin :math:`i`,
- :math:`i_{\mathrm{obs}}` denotes the observed-bin index,
- :math:`j_{\mathrm{true}}` denotes the true-bin index.

The response matrix must satisfy the column-stochastic condition

.. math::

   \sum_i M_{ij} = 1
   \qquad \text{for every true bin } j,

where

- :math:`\sum_i` denotes the sum over all observed bins :math:`i`.

The observed-bin distribution is then

.. math::

   n_i^{\mathrm{obs}}(z)
   =
   \sum_j M_{ij}\, n_j^{\mathrm{true}}(z),

where

- :math:`n_i^{\mathrm{obs}}(z)` is the observed-bin distribution,
- :math:`n_j^{\mathrm{true}}(z)` is the true-bin distribution,
- :math:`M_{ij}` weights how much true bin :math:`j` contributes to observed
  bin :math:`i`.

This means the returned bins may correspond to *observed* tomographic bins,
while still being represented as functions on the common true-redshift grid.


Catastrophic reassignment
~~~~~~~~~~~~~~~~~~~~~~~~~

One supported response effect is catastrophic bin reassignment. For each true
bin :math:`j`, a fraction :math:`f_j` is removed from the diagonal and
redistributed to other bins.

The response can be written schematically as

.. math::

   M_{:j} = (1-f_j)\, e_j + f_j\, q_j,

where

- :math:`M_{:j}` denotes column :math:`j` of the response matrix,
- :math:`f_j` is the catastrophic reassignment fraction for true bin :math:`j`,
- :math:`e_j` is the diagonal basis vector for bin :math:`j`,
- :math:`q_j` is the redistribution pattern over other bins,
- :math:`1-f_j` is the fraction that remains in the original bin.

API mapping:

- :math:`f_j` : ``catastrophic_frac``

Three redistribution models are implemented.


Uniform leakage
^^^^^^^^^^^^^^^

With ``leakage_model="uniform"``, the catastrophic fraction is distributed
equally among all other bins.

This represents nonlocal failures with no preference for nearby bins.


Neighbor leakage
^^^^^^^^^^^^^^^^

With ``leakage_model="neighbor"``, the catastrophic fraction is distributed to
adjacent bins only. Interior bins leak equally to the left and right
neighbors, while edge bins leak to their single available neighbor.

This is a simple model for mostly local misclassification in tomographic-bin
index space.


Gaussian leakage in bin index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``leakage_model="gaussian"``, the catastrophic fraction is redistributed
according to a Gaussian kernel in bin-index space, with width
:math:`\lambda`:

.. math::

   q_j(k) \propto
   \exp\!\left[
      -\frac{(k-j)^2}{2\lambda^2}
   \right]
   \qquad \text{for } k \neq j,

where

- :math:`q_j(k)` is the leakage weight from true bin :math:`j` into bin
  :math:`k`,
- :math:`k` is the receiving bin index,
- :math:`j` is the source true-bin index,
- :math:`\lambda` is the Gaussian leakage width in bin-index space,

API mapping:

- :math:`\lambda` : ``leakage_sigma``

This allows leakage to remain concentrated near the original bin while still
reaching more distant bins with suppressed weight.


Explicit response matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of constructing the catastrophic response from the built-in leakage
models, the user may provide an explicit response matrix directly.

API mapping:

- :math:`M` : ``response_matrix``

When supplied, this matrix overrides the modeled catastrophic reassignment and
is validated to ensure that it has the correct shape and is column-stochastic.

This is useful when the response is known from an external calibration, a mock
catalog, or a survey pipeline rather than a simple parametric model.


Spectroscopic measurement scatter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binny also supports an additional Gaussian measurement-scatter response at the
spec-z level. This differs from catastrophic reassignment: instead of moving a
discrete fraction of objects between bins, it models a continuous measurement
uncertainty in redshift.

A Gaussian spectroscopic measurement model may be written as

.. math::

   \hat{z} \sim \mathcal{N}\!\bigl(z, \sigma_{\mathrm{spec}}(z)\bigr),

where

- :math:`\hat{z}` is the measured spectroscopic redshift,
- :math:`z` is the true redshift,
- :math:`\sigma_{\mathrm{spec}}(z)` is the spectroscopic measurement scatter.

In the implemented parametric form,

.. math::

   \sigma_{\mathrm{spec}}(z) = \sigma_0 + \sigma_1 (1+z),

where

- :math:`\sigma_0` is a constant scatter floor,
- :math:`\sigma_1` is the coefficient of the redshift-dependent scatter term.

API mapping:

- :math:`\sigma_0` : ``sigma0``
- :math:`\sigma_1` : ``sigma1``

At fixed true redshift, the probability of landing in observed bin :math:`i` is

.. math::

   P(i_{\mathrm{obs}} \mid z)
   =
   \int_{z_i}^{z_{i+1}} p(\hat{z}\mid z)\, \mathrm{d}\hat{z},

where

- :math:`P(i_{\mathrm{obs}} \mid z)` is the probability of assigning a galaxy
  at true redshift :math:`z` to observed spectroscopic bin :math:`i`,
- :math:`p(\hat{z}\mid z)` is the conditional density of measured redshift at
  fixed true redshift,
- :math:`z_i` and :math:`z_{i+1}` are the lower and upper edges of observed
  bin :math:`i`,
- :math:`\mathrm{d}\hat{z}` is the integration measure in measured redshift.

The bin-level response matrix is then obtained by averaging this probability
across the true-redshift support of each bin:

.. math::

   M^{\mathrm{scatter}}_{ij}
   \approx
   \left\langle P(i_{\mathrm{obs}} \mid z) \right\rangle_{z \in j},

where

- :math:`M^{\mathrm{scatter}}_{ij}` is the scatter-induced response from true
  bin :math:`j` to observed bin :math:`i`,
- :math:`\langle \cdots \rangle_{z \in j}` denotes an average over the
  true-redshift support of bin :math:`j`.

If catastrophic leakage is also present, the total response becomes

.. math::

   M^{\mathrm{total}} = M^{\mathrm{scatter}} M^{\mathrm{cat}},

where

- :math:`M^{\mathrm{total}}` is the total response matrix,
- :math:`M^{\mathrm{scatter}}` is the measurement-scatter response matrix,
- :math:`M^{\mathrm{cat}}` is the catastrophic-leakage response matrix.

The observed bins are then obtained using :math:`M^{\mathrm{total}}` in place
of :math:`M`.


Implemented spec-z scatter parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two equivalent ways of specifying the spectroscopic scatter are supported.


Explicit per-bin scatter
^^^^^^^^^^^^^^^^^^^^^^^^

The user may provide :math:`\sigma_{\mathrm{spec}}` directly as a scalar or
per-bin sequence.

API mapping:

- :math:`\sigma_{\mathrm{spec}}` : ``specz_scatter``

This sets the Gaussian scatter width associated with each true bin directly.


Parametric scatter model
^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, Binny implements the redshift-dependent model

.. math::

   \sigma_{\mathrm{spec}}(z) = \sigma_0 + \sigma_1 (1+z),

where

- :math:`\sigma_{\mathrm{spec}}(z)` is the measurement scatter,
- :math:`\sigma_0` is the constant floor,
- :math:`\sigma_1` controls redshift growth.

API mapping:

- :math:`\sigma_0` : ``sigma0``
- :math:`\sigma_1` : ``sigma1``

This provides a simple baseline model with both constant and redshift-dependent
contributions.


Implemented spec-z parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following spectroscopic parameters are implemented in Binny:

- :math:`c_j` : ``completeness`` — multiplicative true-bin completeness
- :math:`f_j` : ``catastrophic_frac`` — fraction reassigned away from true bin :math:`j`
- ``leakage_model`` — prescription for redistributing catastrophic leakage
- :math:`\lambda` : ``leakage_sigma`` — width for Gaussian leakage in bin-index space
- :math:`M` : ``response_matrix`` — explicit user-supplied bin-response matrix
- :math:`\sigma_{\mathrm{spec}}` : ``specz_scatter`` — direct Gaussian scatter amplitude
- :math:`\sigma_0` : ``sigma0`` — constant floor in the parametric scatter model
- :math:`\sigma_1` : ``sigma1`` — redshift-dependent coefficient in the parametric scatter model

These components can be used independently or in combination, depending on the
level of realism needed.


No-uncertainty limit
~~~~~~~~~~~~~~~~~~~~

In the simplest spectroscopic limit,

- :math:`c_j = 1` for all bins,
- :math:`f_j = 0` for all bins,
- :math:`\sigma_{\mathrm{spec}} = 0`, or equivalently
  :math:`\sigma_0 = \sigma_1 = 0`,

the response reduces to the identity and each tomographic bin is simply a
top-hat true-redshift slice of the parent distribution.

This is the idealized spectroscopic case with perfectly sharp binning and no
observed-bin mixing.


Normalization and interpretation
--------------------------------

For both photo-z and spec-z builders, Binny distinguishes between the parent
distribution, raw bin weights, and normalized returned bins.


Parent normalization
~~~~~~~~~~~~~~~~~~~~

If ``normalize_input=True``, the parent distribution :math:`n(z)` is first
normalized to integrate to unity over the supplied redshift grid:

.. math::

   \int n(z)\, \mathrm{d}z = 1.

In this equation,

- :math:`n(z)` is the parent redshift distribution,
- :math:`\mathrm{d}z` is the integration measure in true redshift.

This ensures that bin construction begins from a properly normalized parent
probability density.


Bin normalization
~~~~~~~~~~~~~~~~~

If ``normalize_bins=True``, each individual tomographic bin is normalized after
construction. The normalized bin may then be written as

.. math::

   \tilde{n}_i(z)
   =
   \frac{n_i(z)}
        {\int n_i(z')\, \mathrm{d}z'},

where

- :math:`\tilde{n}_i(z)` is the normalized version of bin :math:`i`,
- :math:`n_i(z)` is the unnormalized bin distribution,
- :math:`z'` is a dummy integration variable,
- :math:`\mathrm{d}z'` is the integration measure with respect to :math:`z'`.

This makes the returned bins easy to compare as redshift distributions, but it
also removes their relative population amplitudes from the returned arrays.

As a result:

- the returned bins encode **shape**;
- the metadata encode **population fractions**.

This separation is often useful in cosmological workflows, where one may want
normalized kernel shapes together with external number-density information or
internally stored fractional weights.


Summary
-------

Binny separates two physically distinct redshift-selection problems.

For **photo-z tomography**, observed-redshift bins are mapped onto the
true-redshift grid through a probabilistic assignment model,

.. math::

   n_i(z) = n(z)\, P(i \mid z),

where all symbols were defined above. The implemented uncertainty terms include
core Gaussian scatter, additive and multiplicative bias terms, and an optional
outlier Gaussian mixture.

For **spec-z tomography**, true-redshift bins are first defined directly through
a deterministic selection model and may then be modified by completeness,
catastrophic bin reassignment, explicit response matrices, and Gaussian
measurement scatter.

Together, these models provide a flexible framework for constructing
tomographic redshift distributions whose uncertainty assumptions remain
explicit, interpretable, and easy to validate.