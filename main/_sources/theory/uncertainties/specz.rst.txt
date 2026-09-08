.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Spectroscopic-redshift uncertainties
===========================================

In spectroscopic-redshift tomography, bins are defined directly in true
redshift. In the ideal case this corresponds to perfectly sharp bin edges.
Real surveys, however, may introduce incompleteness, misclassification, or
measurement scatter, which can be modeled through additional response terms.

In the simplest case, a tomographic bin is a top-hat selection over a
true redshift interval.

For true redshift edges :math:`[z_j, z_{j+1}]`, the true-bin selection window is

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),

where

- :math:`S_j(z)` is the true redshift selection function for true bin :math:`j`,
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


Completeness
------------

The simplest spec-z uncertainty ingredient is the completeness factor:

.. math::

   S_j(z) = c_j\, \mathbf{1}_{[z_j, z_{j+1})}(z),
   \qquad 0 \le c_j \le 1,

where

- :math:`c_j=1` corresponds to full completeness in bin :math:`j`,
- :math:`c_j<1` reduces the pre-normalization population in that bin.

This parameter represents objects that should belong to the bin but are
missing from the sample, for example due to survey selection
effects, failed spectroscopic measurements, or masking.

Because completeness acts multiplicatively, it changes bin populations before
optional normalization. If bins are later normalized individually, the shape
of a nonempty bin is preserved while its **total population decreases**.

In other words, completeness does not move galaxies between bins.
It simply removes a fraction of galaxies from the affected bin.

The animation below illustrates this effect: as completeness decreases,
the height of the affected bin decreases while the other bins remain unchanged.

.. image:: ../../_static/animations/specz_uncertainty_completeness.gif
   :alt: Animation showing the effect of decreasing completeness in a spectroscopic tomographic bin
   :width: 400px
   :align: center

API mapping:

- :math:`c_j` : ``completeness``


Observed-bin response
---------------------

Although spec-z bins are defined in true redshift, Binny also supports response
effects that map true bins into observed bins. This is useful for modeling bin
reassignment, imperfect classification, or leakage between bins.

The observed-bin distributions are constructed from the true-bin distributions
through a response matrix :math:`M`:

.. math::

   M_{ij} = P(i_{\mathrm{obs}} \mid j_{\mathrm{true}}),

where

- :math:`M_{ij}` is the probability that an object in true bin :math:`j`
  appears in observed bin :math:`i`,
- :math:`i_{\mathrm{obs}}` denotes the observed-bin index,
- :math:`j_{\mathrm{true}}` denotes the true-bin index.

The response matrix must satisfy the column-stochastic condition

.. math::

   \sum_i M_{ij} = 1
   \qquad \text{for every true bin } j,

which simply means that **every galaxy must end up in some observed bin**.

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

The response matrix therefore determines how the true-bin populations are
distributed across the observed bins. Instead of each true bin contributing
only to its corresponding observed bin, part of its population may be
redistributed to other bins.


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

In practice this parameter represents **objects that are assigned to the wrong
tomographic bin**. Increasing :math:`f_j` moves a larger fraction of galaxies
from their true bin into other bins.

API mapping:

- :math:`f_j` : ``catastrophic_frac``


Uniform leakage
^^^^^^^^^^^^^^^

With ``leakage_model="uniform"``, the catastrophic fraction is distributed
equally among all other bins.

This represents situations where misclassified galaxies are scattered
randomly across all bins, with no preference for nearby bins.


Neighbor leakage
^^^^^^^^^^^^^^^^

With ``leakage_model="neighbor"``, the catastrophic fraction is distributed to
adjacent bins only. Interior bins leak equally to the left and right
neighbors, while edge bins leak to their single available neighbor.

This is a simple model for local misclassification, where galaxies are most
likely to be assigned to bins that are close in redshift.


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
- :math:`\lambda` is the Gaussian leakage width in bin-index space.

API mapping:

- :math:`\lambda` : ``leakage_sigma``

This allows leakage to remain concentrated near the original bin while still
reaching more distant bins with suppressed weight.

.. image:: ../../_static/animations/specz_uncertainty_catastrophic.gif
   :alt: Animation showing catastrophic spectroscopic bin reassignment with neighbor leakage
   :width: 800px
   :align: center

The animation above shows the **neighbor-leakage case**. As the catastrophic
fraction :math:`f_2` increases, a larger fraction of galaxies from the second
true bin is reassigned to its neighboring bins. The left panel shows the
resulting observed-bin distributions, while the matrix on the right shows the
corresponding response matrix.


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
---------------------------------

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
across the true redshift support of each bin:

.. math::

   M^{\mathrm{scatter}}_{ij}
   \approx
   \left\langle P(i_{\mathrm{obs}} \mid z) \right\rangle_{z \in j},

where

- :math:`M^{\mathrm{scatter}}_{ij}` is the scatter-induced response from true
  bin :math:`j` to observed bin :math:`i`,
- :math:`\langle \cdots \rangle_{z \in j}` denotes an average over the
  true redshift support of bin :math:`j`.

If catastrophic leakage is also present, the total response becomes

.. math::

   M^{\mathrm{total}} = M^{\mathrm{scatter}} M^{\mathrm{cat}},

where

- :math:`M^{\mathrm{total}}` is the total response matrix,
- :math:`M^{\mathrm{scatter}}` is the measurement-scatter response matrix,
- :math:`M^{\mathrm{cat}}` is the catastrophic-leakage response matrix.

The observed bins are then obtained using :math:`M^{\mathrm{total}}` in place
of :math:`M`.

.. image:: ../../_static/animations/specz_uncertainty_scatter.gif
   :alt: Animation showing the effect of increasing spectroscopic measurement scatter on tomographic bins
   :width: 400px
   :align: center

API mapping:

- :math:`\sigma_0` : ``sigma0``
- :math:`\sigma_1` : ``sigma1``


Scatter parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~

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
-----------------------------

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
--------------------

In the simplest spectroscopic limit,

- :math:`c_j = 1` for all bins,
- :math:`f_j = 0` for all bins,
- :math:`\sigma_{\mathrm{spec}} = 0`, or equivalently
  :math:`\sigma_0 = \sigma_1 = 0`,

the response reduces to the identity and each tomographic bin is simply a
top-hat true redshift slice of the parent distribution.

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
