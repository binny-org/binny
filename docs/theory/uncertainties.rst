.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Redshift uncertainty models
==================================

This page describes the redshift-selection models currently implemented in
Binny for both photometric-redshift (**photo-z**) and spectroscopic-redshift
(**spec-z**) tomography.

The goal is to make explicit how tomographic bins are constructed from an
underlying parent redshift distribution :math:`n(z)`, how uncertainties enter
the selection, and how those uncertainties modify the final bin distributions
used downstream in diagnostics or forecasting.

Binny supports two broad cases:

- **Photo-z binning**, where bins are defined in observed redshift and mapped
  back onto the true-redshift grid through a probabilistic assignment model.
- **Spec-z binning**, where bins are defined in true redshift, with optional
  completeness losses and bin-to-bin response effects that model residual
  spectroscopic failures or measurement scatter.

In both cases, the returned tomographic bins are evaluated on a common
true-redshift grid :math:`z`.


General framework
-----------------

We begin from a parent redshift distribution :math:`n(z)`, defined on a
true-redshift grid. A tomographic construction produces a set of bin
distributions

.. math::

   n_i(z),

where :math:`i` labels the tomographic bin.

More generally, the tomographic construction can be written as

.. math::

   n_i(z) = n(z)\, S_i(z),

where :math:`S_i(z)` is the effective selection function for bin :math:`i`.
For spec-z binning, :math:`S_i(z)` is typically a top-hat-like selection in
true redshift. For photo-z binning, this role is played by the bin-assignment
probability :math:`P(i \mid z)`.

Conceptually, each bin is obtained by applying a redshift-selection model to
the parent population. The main difference between the photo-z and spec-z
implementations is where the bin edges live and how uncertainty is modeled:

- in the **photo-z** case, bin edges live in observed-redshift space and the
  binning is probabilistic;
- in the **spec-z** case, bin edges live in true-redshift space and the
  binning is primarily deterministic, with optional response effects layered on
  top.

If requested, Binny also records metadata describing pre-normalization bin
weights, so the user can distinguish between bin *shape* and bin *population*.


Photometric-redshift model
--------------------------

In photometric-redshift tomography, the tomographic bins are defined in
observed redshift, but the returned bin distributions are evaluated on the
true-redshift grid. For a photo-z bin with observed-redshift edges
:math:`[z_{\mathrm{ph,min}}, z_{\mathrm{ph,max}}]`, the selected
true-redshift distribution is

.. math::

   n_i(z) = n(z)\, P(i \mid z),

where :math:`P(i \mid z)` is the probability that an object at true redshift
:math:`z` is assigned to observed bin :math:`i`.

This means that photo-z tomography does not apply a hard cut in true
redshift. Instead, each true-redshift value contributes to a bin according to
the photo-z error model.


Core Gaussian assignment
~~~~~~~~~~~~~~~~~~~~~~~~

The implemented core photo-z model assumes that the observed redshift
:math:`z_{\mathrm{ph}}` at fixed true redshift :math:`z` is Gaussian:

.. math::

   z_{\mathrm{ph}} \sim \mathcal{N}(\mu(z), \sigma(z)),

with

.. math::

   \mu(z) = \mathrm{mean\_scale}\, z - \mathrm{mean\_offset},

.. math::

   \sigma(z) = \mathrm{scatter\_scale}\, (1 + z).

The corresponding conditional density is

.. math::

   p(z_{\mathrm{ph}} \mid z)
   =
   \frac{1}{\sqrt{2\pi}\,\sigma(z)}
   \exp\!\left[
      -\frac{\bigl(z_{\mathrm{ph}}-\mu(z)\bigr)^2}{2\sigma^2(z)}
   \right].

The bin-assignment probability is then the Gaussian probability integrated
between the observed bin edges:

.. math::

   P(i \mid z)
   =
   \int_{z_{\mathrm{ph,min},i}}^{z_{\mathrm{ph,max},i}}
   p(z_{\mathrm{ph}} \mid z)\, dz_{\mathrm{ph}}.

Because the model is Gaussian, this integral can be evaluated analytically as

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
   \right].

In the implementation, this analytic form is used directly, making the
assignment smooth, fast, and numerically stable.


No-uncertainty limit
~~~~~~~~~~~~~~~~~~~~

If :math:`\mathrm{scatter\_scale} = 0`, the Gaussian model reduces to a
deterministic mapping. In that limit, the observed redshift is simply

.. math::

   z_{\mathrm{ph}} = \mu(z),

and the bin-assignment probability becomes a top-hat indicator in photo-z
space:

.. math::

   P(i \mid z) =
   \begin{cases}
   1, & z_{\mathrm{ph}} \in [z_{\mathrm{ph,min},i}, z_{\mathrm{ph,max},i}) \\
   0, & \text{otherwise.}
   \end{cases}

This provides a useful limiting case where photo-z binning becomes a sharp
selection under the chosen bias convention.


Photo-z uncertainties included in Binny
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented photo-z model includes several distinct uncertainty terms.


Scatter
^^^^^^^

The parameter ``scatter_scale`` controls the width of the Gaussian photo-z
distribution,

.. math::

   \sigma(z) = \mathrm{scatter\_scale}\, (1+z).

This describes the random broadening of the photo-z estimate around its mean
relation. Larger values lead to broader bin-assignment probabilities, more
overlap between neighboring tomographic bins, and less sharply localized
true-redshift distributions.


Bias / mean shift
^^^^^^^^^^^^^^^^^

The parameter ``mean_offset`` enters the mean relation as

.. math::

   \mu(z) = \mathrm{mean\_scale}\, z - \mathrm{mean\_offset}.

This shifts the effective photo-z estimate relative to the true redshift and
therefore models a systematic redshift bias. Positive or negative offsets move
the mapping between true-z and observed-z bins and can shift the apparent
location of the tomographic selection.


Mean scaling
^^^^^^^^^^^^

The parameter ``mean_scale`` multiplies the true redshift in the mean relation.
This allows a linear distortion of the mean photo-z mapping rather than a
simple additive shift:

.. math::

   \mu(z) = \mathrm{mean\_scale}\, z - \mathrm{mean\_offset}.

When :math:`\mathrm{mean\_scale} \neq 1`, the photo-z relation can stretch or
compress with redshift, representing a calibration mismatch that grows with
:math:`z`.


Outlier mixture
^^^^^^^^^^^^^^^

Binny also supports a second Gaussian component representing catastrophic or
outlier-like photo-z failures. In that case, the conditional density is
written as a two-component mixture,

.. math::

   p(z_{\mathrm{ph}} \mid z)
   =
   (1-f_{\mathrm{out}})\,
   p_{\mathrm{core}}(z_{\mathrm{ph}} \mid z)
   +
   f_{\mathrm{out}}\,
   p_{\mathrm{out}}(z_{\mathrm{ph}} \mid z),

where :math:`f_{\mathrm{out}}` is the parameter ``outlier_frac``.

Equivalently, the bin-assignment probability becomes

.. math::

   P(i \mid z)
   =
   (1-f_{\mathrm{out}})\, P_{\mathrm{core}}(i \mid z)
   +
   f_{\mathrm{out}}\, P_{\mathrm{out}}(i \mid z).

The outlier component has its own mean and scatter model:

.. math::

   \mu_{\mathrm{out}}(z)
   =
   \mathrm{outlier\_mean\_scale}\, z
   -
   \mathrm{outlier\_mean\_offset},

.. math::

   \sigma_{\mathrm{out}}(z)
   =
   \mathrm{outlier\_scatter\_scale}\, (1+z).

This allows the model to represent a minority population whose photo-z values
are distributed differently from the core sample, for example broader,
shifted, or otherwise misassigned objects.


Implemented photo-z parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following uncertainty-related parameters are implemented in the photo-z
builder:

- ``scatter_scale``: core Gaussian scatter amplitude
- ``mean_offset``: additive shift in the core mean relation
- ``mean_scale``: multiplicative scaling in the core mean relation
- ``outlier_frac``: fraction of objects assigned to the outlier component
- ``outlier_scatter_scale``: scatter amplitude of the outlier Gaussian
- ``outlier_mean_offset``: additive shift of the outlier mean relation
- ``outlier_mean_scale``: multiplicative scaling of the outlier mean relation

Each of these may be provided either as a scalar, applying the same value to
all bins, or as a per-bin sequence.


Interpreting the returned photo-z bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The returned photo-z tomographic bins are distributions on the **true**
redshift grid. They should therefore be read as

.. math::

   n_i(z) = n(z)\, P(i \mid z),

not as histograms directly in observed-redshift space.

If ``normalize_bins=True``, each bin is normalized to integrate to unity and
therefore becomes a shape-only redshift probability density. In that case, the
relative population fractions of the bins are not encoded in the returned
curves themselves and should instead be taken from the metadata.


Metadata and population fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If metadata is requested, Binny records how much of the original parent
distribution falls into each observed bin **before** any per-bin
normalization.

The stored quantities include:

- ``parent_norm``: total area under the parent :math:`n(z)`
- ``bins_norms[i]``: area under the raw bin curve before normalization
- ``frac_per_bin[i]``: the fractional population in bin :math:`i`, computed as
  ``bins_norms[i] / parent_norm`` when the parent norm is nonzero

This distinction is important: once individual bins are normalized, they retain
their redshift *shape* but no longer retain their relative *abundance*.


Spectroscopic-redshift model
----------------------------

In spectroscopic-redshift tomography, the bins are defined directly in
true-redshift space. In the simplest case, a tomographic bin is just a
top-hat selection over a true-redshift interval.

For true-redshift edges :math:`[z_j, z_{j+1}]`, the true-bin selection window
is

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),

where :math:`c_j` is a completeness factor and :math:`\mathbf{1}` is the
indicator function.

The corresponding true-bin distribution is

.. math::

   n_j^{\mathrm{true}}(z) = n(z)\, S_j(z).

In the absence of any response effects, the observed and true bins coincide, so

.. math::

   n_i^{\mathrm{obs}}(z) = n_i^{\mathrm{true}}(z).

In this sense, spec-z tomography starts from a deterministic true-redshift
partition, unlike photo-z tomography which is probabilistic from the outset.


Completeness uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~

The first uncertainty-like ingredient in the spec-z model is the
``completeness`` parameter. This multiplies the selection window in each bin:

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),
   \qquad 0 \le c_j \le 1.

This allows the user to represent incomplete sampling or bin-dependent
spectroscopic success rates. A completeness of unity leaves the bin unchanged,
while smaller values reduce the pre-normalization weight of that bin.

Because completeness acts multiplicatively, it changes bin populations before
optional normalization. If bins are later normalized individually, the shape of
a nonempty bin is preserved but its reduced population is only visible through
metadata.


Observed-bin response
~~~~~~~~~~~~~~~~~~~~~

Although spec-z bins are defined in true redshift, Binny also supports survey
response effects that map true bins into observed bins. This is useful when one
wants to model bin reassignment, imperfect classification, or small
measurement-induced leakage between bins.

The observed-bin distributions are constructed from the true-bin distributions
through a response matrix :math:`M`:

.. math::

   M[i,j] = P(i_{\mathrm{obs}} \mid j_{\mathrm{true}}),

with the column-stochastic condition

.. math::

   \sum_i M[i,j] = 1
   \qquad \text{for every true bin } j.

The observed-bin distribution is then

.. math::

   n_i^{\mathrm{obs}}(z)
   =
   \sum_j M[i,j]\, n_j^{\mathrm{true}}(z).

This means the returned bins may correspond to *observed* tomographic bins,
while still being represented as functions on the common true-redshift grid.


Catastrophic reassignment
~~~~~~~~~~~~~~~~~~~~~~~~~

One supported response effect is catastrophic bin reassignment, controlled by
``catastrophic_frac``. For each true bin :math:`j`, a fraction
:math:`f_j` is taken away from the diagonal and redistributed to other bins.

This produces a response of the schematic form

.. math::

   M[:,j] = (1-f_j)\, e_j + f_j\, q_j,

where :math:`e_j` is the diagonal basis vector and :math:`q_j` is a
redistribution pattern over other bins.

Three redistribution models are implemented.


Uniform leakage
^^^^^^^^^^^^^^^

With ``leakage_model="uniform"``, the catastrophic fraction is distributed
equally among all other bins. This represents nonlocal failures with no
preference for nearby bins.


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
according to a Gaussian kernel in bin-index space, with width controlled by
``leakage_sigma``.

This allows leakage to be concentrated near the original bin while still
reaching more distant bins with suppressed weight.


Explicit response matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of constructing the catastrophic response from the built-in leakage
models, the user may provide an explicit ``response_matrix`` directly.

When supplied, this matrix overrides the modeled catastrophic reassignment and
is validated to ensure that it has the correct shape and is column-stochastic.

This is useful when the response is known from an external calibration, a mock
catalog, or a survey pipeline rather than a simple parametric model.


Spectroscopic measurement scatter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binny also supports an additional Gaussian measurement-scatter response at the
spec-z level. This is conceptually different from catastrophic reassignment:
instead of redistributing a discrete fraction of objects between bins, it
models a continuous measurement uncertainty in redshift.

A Gaussian spectroscopic measurement model may be written as

.. math::

   \hat{z} \sim \mathcal{N}(z, \sigma_{\mathrm{spec}}(z)),

with, in the implemented parametric form,

.. math::

   \sigma_{\mathrm{spec}}(z) = \sigma_0 + \sigma_1 (1+z).

At fixed true redshift, the probability of landing in observed bin
:math:`i` is

.. math::

   P(i_{\mathrm{obs}} \mid z)
   =
   \int_{z_i}^{z_{i+1}} p(\hat{z}\mid z)\, d\hat{z}.

The bin-level response matrix is then obtained by averaging this probability
across the true-redshift support of each bin:

.. math::

   M_{\mathrm{scatter}}[i,j]
   \approx
   \left\langle P(i_{\mathrm{obs}} \mid z) \right\rangle_{z \in j}.

If enabled, this yields a scatter matrix :math:`M_{\mathrm{scatter}}`. When
combined with catastrophic leakage, the total response is

.. math::

   M_{\mathrm{total}} = M_{\mathrm{scatter}}\, M_{\mathrm{cat}}.

The observed bins are then obtained from :math:`M_{\mathrm{total}}`.


Implemented spec-z scatter parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two equivalent ways of specifying the spectroscopic scatter are supported.


Explicit per-bin scatter
^^^^^^^^^^^^^^^^^^^^^^^^

The user may provide ``specz_scatter`` directly as a scalar or per-bin
sequence. This sets the Gaussian width associated with each true bin.


Parametric scatter model
^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, Binny implements a simple redshift-dependent model,

.. math::

   \sigma(z) = \sigma_0 + \sigma_1 (1+z),

through the parameters ``sigma0`` and ``sigma1``.

This gives a flexible baseline model in which the scatter can include both a
constant floor and a term that grows with redshift.


Spec-z uncertainties included in Binny
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented spectroscopic model includes the following uncertainty or
response ingredients:

- ``completeness``: multiplicative true-bin completeness
- ``catastrophic_frac``: fraction reassigned away from the true bin
- ``leakage_model``: prescription for redistributing catastrophic leakage
- ``leakage_sigma``: width for Gaussian leakage in bin-index space
- ``response_matrix``: explicit user-supplied bin-response matrix
- ``specz_scatter``: direct Gaussian scatter amplitude
- ``sigma0`` and ``sigma1``: parameters for a redshift-dependent Gaussian
  scatter model

These components can be used independently or in combination, depending on the
level of realism needed.


No-uncertainty limit
~~~~~~~~~~~~~~~~~~~~

In the simplest spec-z limit,

- ``completeness = 1``,
- ``catastrophic_frac = 0``,
- ``specz_scatter = 0`` or ``sigma0 = sigma1 = 0``,

the response reduces to the identity and each tomographic bin is simply a
top-hat true-redshift slice of the parent distribution.

This provides the idealized spectroscopic case with perfectly sharp binning and
no observed-bin mixing.


Normalization and interpretation
--------------------------------

For both photo-z and spec-z builders, Binny distinguishes between the parent
distribution, raw bin weights, and normalized returned bins.


Parent normalization
~~~~~~~~~~~~~~~~~~~~

If ``normalize_input=True``, the parent distribution :math:`n(z)` is first
normalized to integrate to unity over the supplied grid. This ensures that the
bin construction is based on a properly normalized parent probability density.


Bin normalization
~~~~~~~~~~~~~~~~~

If ``normalize_bins=True``, each individual tomographic bin is normalized to
integrate to unity after construction. In that case the normalized bin can be
written as

.. math::

   \tilde{n}_i(z)
   =
   \frac{n_i(z)}
        {\int n_i(z')\, dz'},

provided the bin has nonzero support.

This makes the output bins easy to compare as redshift distributions, but it
also removes their relative population amplitudes from the returned arrays.

As a result:

- the returned bins encode **shape**;
- the metadata encodes **population fractions**.

This separation is often useful in cosmological workflows, where one may want
normalized kernel shapes together with external number-density information or
internally stored fractional weights.


Summary
-------

The implemented Binny tomography models separate two physically distinct cases.

For **photo-z tomography**, observed-redshift bins are mapped onto the
true-redshift grid through a probabilistic assignment model. The implemented
uncertainties include Gaussian scatter, mean bias, mean scaling, and an
optional outlier Gaussian mixture.

For **spec-z tomography**, true-redshift bins are defined directly and may then
be modified by incompleteness, catastrophic bin reassignment, explicit
response matrices, and Gaussian measurement scatter.

Together, these models provide a flexible framework for building tomographic
redshift distributions with uncertainty models that remain explicit,
interpretable, and easy to validate.