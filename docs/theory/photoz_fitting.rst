.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Photometric-redshift error fitting
=========================================

Photometric-redshift uncertainties affect the assignment of galaxies to
tomographic bins and therefore the redshift distributions used in cosmological
analyses. These uncertainties are commonly described using a statistical model
for the relationship between true and observed redshifts.

Binny provides an unbinned maximum-likelihood method for estimating
photometric-redshift uncertainty parameters from catalogues containing matched
true and observed redshifts.

The method fits a two-component Gaussian mixture to normalized photo-z
residuals. The recovered parameters can be converted into Binny's forward
photo-z uncertainty model and used to construct tomographic redshift
distributions.

For executable examples, see
:doc:`../examples/tomography/photoz_fitting`.


Normalized photometric-redshift residuals
-----------------------------------------

For a galaxy with true redshift :math:`z` and observed photometric redshift
:math:`z_{\mathrm{ph}}`, define the normalized residual

.. math::

   \epsilon =
   \frac{z_{\mathrm{ph}} - z}{1+z}.

This normalization accounts for the approximate redshift dependence of
photometric-redshift scatter, which is often modelled as proportional to
:math:`1+z`.

The observed redshift can equivalently be expressed as

.. math::

   z_{\mathrm{ph}} = z + (1+z)\epsilon.

The fitting model assumes that the distribution of normalized residuals is
independent of true redshift over the catalogue being fitted.

This provides a compact description of the photo-z error distribution using
parameters shared across the sample.


Gaussian-mixture model
----------------------

Photometric-redshift errors often contain a dominant population of relatively
accurate estimates together with a smaller population of objects exhibiting
larger errors or systematic displacements.

Binny models these populations using a two-component Gaussian mixture,

.. math::

   p(\epsilon \mid \boldsymbol{\phi})
   =
   (1-f_{\mathrm{out}})
   \mathcal{N}(\epsilon;0,\sigma_{\mathrm{core}}^2)
   +
   f_{\mathrm{out}}
   \mathcal{N}(\epsilon;\mu_{\mathrm{out}},\sigma_{\mathrm{out}}^2),

where

.. math::

   \boldsymbol{\phi}
   =
   \left(
   \sigma_{\mathrm{core}},
   \sigma_{\mathrm{out}},
   \mu_{\mathrm{out}},
   f_{\mathrm{out}}
   \right).

The Gaussian probability density is defined as

.. math::

   \mathcal{N}(\epsilon;\mu,\sigma^2)
   =
   \frac{1}{\sqrt{2\pi}\sigma}
   \exp\left[
   -\frac{(\epsilon-\mu)^2}{2\sigma^2}
   \right].

The parameters describe:

- :math:`\sigma_{\mathrm{core}}`: scatter of the main photo-z population;
- :math:`\sigma_{\mathrm{out}}`: scatter of the outlier population;
- :math:`\mu_{\mathrm{out}}`: mean normalized residual of the outlier population;
- :math:`f_{\mathrm{out}}`: fraction assigned to the outlier component.

The core component has a mean fixed to zero, while the outlier component has
an independent mean and scatter.

The mixture weights satisfy

.. math::

   0 \leq f_{\mathrm{out}} \leq 1,
   \qquad
   (1-f_{\mathrm{out}})+f_{\mathrm{out}}=1.

To identify the dominant population as the core, Binny imposes a minimum
core weight,

.. math::

   1-f_{\mathrm{out}}
   \geq w_{\mathrm{core,min}}.

The default value is :math:`w_{\mathrm{core,min}}=0.5`, restricting the fitted
outlier fraction to at most :math:`0.5`.

This convention reduces ambiguity in component labelling but does not
guarantee that the two populations are statistically distinguishable.


Unbinned maximum-likelihood estimation
--------------------------------------

Consider a catalogue containing :math:`N` matched true and observed redshifts,

.. math::

   \mathcal{D}
   =
   \left\{
   (z_i,z_{\mathrm{ph},i})
   \right\}_{i=1}^{N}.

The normalized residual for each galaxy is

.. math::

   \epsilon_i =
   \frac{z_{\mathrm{ph},i}-z_i}{1+z_i}.

Assuming independent residuals, the likelihood of the mixture parameters is

.. math::

   \mathcal{L}(\boldsymbol{\phi})
   =
   \prod_{i=1}^{N}
   p(\epsilon_i\mid\boldsymbol{\phi}).

The corresponding log-likelihood is

.. math::

   \ln\mathcal{L}(\boldsymbol{\phi})
   =
   \sum_{i=1}^{N}
   \ln\left[
   (1-f_{\mathrm{out}})
   \mathcal{N}(\epsilon_i;0,\sigma_{\mathrm{core}}^2)
   +
   f_{\mathrm{out}}
   \mathcal{N}(\epsilon_i;\mu_{\mathrm{out}},
   \sigma_{\mathrm{out}}^2)
   \right].

The maximum-likelihood estimator is

.. math::

   \hat{\boldsymbol{\phi}}
   =
   \underset{\boldsymbol{\phi}}{\operatorname{argmax}}
   \;
   \ln\mathcal{L}(\boldsymbol{\phi}),

subject to the mixture-weight constraint and positive component scatters.

The likelihood is evaluated directly from individual galaxy residuals,
without first constructing a histogram.

Consequently, the fitted parameters do not depend on the choice of histogram
bin edges.


Expectation-maximization algorithm
----------------------------------

Binny estimates the Gaussian-mixture parameters using the
expectation-maximization (EM) algorithm.

The method introduces a latent component label :math:`c_i\in\{0,1\}` for
each galaxy, with :math:`c_i=0` denoting the core and :math:`c_i=1`
denoting the outlier population.

EM alternates between estimating the probability that each galaxy belongs
to the outlier component and updating the mixture parameters.


Expectation step
^^^^^^^^^^^^^^^^

The posterior probability that galaxy :math:`i` belongs to the outlier
component is

.. math::

   r_i
   =
   \frac{
   f_{\mathrm{out}}
   \mathcal{N}(\epsilon_i;\mu_{\mathrm{out}},
   \sigma_{\mathrm{out}}^2)
   }{
   (1-f_{\mathrm{out}})
   \mathcal{N}(\epsilon_i;0,\sigma_{\mathrm{core}}^2)
   +
   f_{\mathrm{out}}
   \mathcal{N}(\epsilon_i;\mu_{\mathrm{out}},
   \sigma_{\mathrm{out}}^2)
   }.

The quantity :math:`r_i` is called the outlier responsibility.

The corresponding probability of core membership is

.. math::

   1-r_i.

These probabilities provide a soft assignment of galaxies to the two
mixture components.


Maximization step
^^^^^^^^^^^^^^^^^

Define the effective component populations as

.. math::

   N_{\mathrm{out}} = \sum_{i=1}^{N}r_i,
   \qquad
   N_{\mathrm{core}} = \sum_{i=1}^{N}(1-r_i).

The unconstrained update for the outlier fraction is

.. math::

   f_{\mathrm{out}}^{\mathrm{new}}
   =
   \frac{N_{\mathrm{out}}}{N}.

The outlier mean is updated according to

.. math::

   \mu_{\mathrm{out}}^{\mathrm{new}}
   =
   \frac{\sum_i r_i\epsilon_i}{N_{\mathrm{out}}}.

The core variance update is

.. math::

   \left(\sigma_{\mathrm{core}}^2\right)^{\mathrm{new}}
   =
   \frac{
   \sum_i(1-r_i)\epsilon_i^2
   }{
   N_{\mathrm{core}}
   }.

The core mean remains fixed at zero, so no fitted mean is subtracted in
this expression.

The outlier variance update is

.. math::

   \left(\sigma_{\mathrm{out}}^2\right)^{\mathrm{new}}
   =
   \frac{
   \sum_i r_i
   \left(\epsilon_i-\mu_{\mathrm{out}}^{\mathrm{new}}\right)^2
   }{
   N_{\mathrm{out}}
   }.

The core-weight constraint restricts the updated outlier fraction to

.. math::

   f_{\mathrm{out}}
   \leq 1-w_{\mathrm{core,min}}.

The algorithm repeats the expectation and maximization steps until its
convergence criterion is satisfied or the maximum number of iterations
is reached.

Gaussian-mixture likelihoods may contain multiple local optima.
Binny therefore supports multiple initializations through ``n_init``.
The ``seed`` argument controls the reproducibility of these
initializations.


Connection to the forward photo-z model
---------------------------------------

The fitted Gaussian mixture describes normalized residuals, while
tomographic binning requires the conditional distribution of observed
photometric redshift at fixed true redshift,

.. math::

   p(z_{\mathrm{ph}}\mid z).

The transformation between these quantities follows from

.. math::

   z_{\mathrm{ph}}=z+(1+z)\epsilon.

For a Gaussian residual component with mean :math:`\mu` and standard
deviation :math:`\sigma`, the corresponding conditional distribution is

.. math::

   z_{\mathrm{ph}}\mid z
   \sim
   \mathcal{N}
   \left(
   z+(1+z)\mu,
   (1+z)^2\sigma^2
   \right).

The core component has zero residual mean, giving

.. math::

   \langle z_{\mathrm{ph}}\mid z\rangle_{\mathrm{core}}=z,

and

.. math::

   \sigma_{z,\mathrm{core}}(z)
   =
   (1+z)\sigma_{\mathrm{core}}.

For the outlier component,

.. math::

   \langle z_{\mathrm{ph}}\mid z\rangle_{\mathrm{out}}
   =
   z+(1+z)\mu_{\mathrm{out}},

which can be rewritten as

.. math::

   \langle z_{\mathrm{ph}}\mid z\rangle_{\mathrm{out}}
   =
   (1+\mu_{\mathrm{out}})z+\mu_{\mathrm{out}}.

Binny provides
:meth:`binny.NZTomography.to_binny_photoz_params`
to translate the fitted residual parameters into the uncertainty
parameters accepted by the forward photo-z assignment model.

The implemented conversion is

.. math::

   \begin{aligned}
   \texttt{scatter_scale}
   &= \sigma_{\mathrm{core}},\\
   \texttt{outlier_scatter_scale}
   &= \sigma_{\mathrm{out}},\\
   \texttt{outlier_frac}
   &= f_{\mathrm{out}},\\
   \texttt{outlier_mean_scale}
   &= 1+\mu_{\mathrm{out}},\\
   \texttt{outlier_mean_offset}
   &= -\mu_{\mathrm{out}}.
   \end{aligned}

The core mean parameters retain their default values,

.. math::

   \texttt{mean\_scale}=1,
   \qquad
   \texttt{mean\_offset}=0.

The offset sign follows Binny's forward-model parameter convention.
The conversion should be interpreted together with that convention,
rather than as a direct identification of the offset parameter with
the observed residual mean.

The resulting parameters can be passed to the ``uncertainties`` field
of a photometric tomography specification.


Tomographic assignment probabilities
------------------------------------

Let :math:`n(z)` denote the parent true-redshift distribution.

A photometric tomographic bin :math:`i` is defined by observed-redshift
boundaries :math:`[z_i^{\mathrm{min}},z_i^{\mathrm{max}}]`.

The probability that a galaxy at true redshift :math:`z` is assigned
to this bin is

.. math::

   P_i(z)
   =
   \int_{z_i^{\mathrm{min}}}^{z_i^{\mathrm{max}}}
   p(z_{\mathrm{ph}}\mid z)
   \,\mathrm{d}z_{\mathrm{ph}}.

For a Gaussian component with conditional mean :math:`m(z)` and
standard deviation :math:`s(z)`, this integral can be expressed using
the standard normal cumulative distribution function :math:`\Phi`,

.. math::

   P_i(z)
   =
   \Phi\left(
   \frac{z_i^{\mathrm{max}}-m(z)}{s(z)}
   \right)
   -
   \Phi\left(
   \frac{z_i^{\mathrm{min}}-m(z)}{s(z)}
   \right).

For the two-component photo-z mixture, the full assignment probability is

.. math::

   P_i(z)
   =
   (1-f_{\mathrm{out}})
   P_{i,\mathrm{core}}(z)
   +
   f_{\mathrm{out}}
   P_{i,\mathrm{out}}(z).

The resulting tomographic true-redshift distribution is

.. math::

   n_i(z)=n(z)P_i(z).

This is the same general window-function construction used throughout
Binny's tomography framework, with

.. math::

   W_i(z)=P_i(z).

The fitted photo-z uncertainty model therefore determines the
probabilistic window functions used to construct the tomographic bins.

Depending on the configuration, each resulting distribution may be
normalized independently,

.. math::

   \int n_i(z)\,\mathrm{d}z=1.

For further discussion of tomographic construction and normalization,
see :doc:`tomography`.


Fit diagnostics
---------------

Binny optionally returns statistical and numerical diagnostics when
``return_metadata=True``.

The metadata include the log-likelihood, convergence status, number of
iterations, information criteria, and a goodness-of-fit statistic.

The Akaike information criterion is

.. math::

   \mathrm{AIC}
   =
   2k-2\ln\hat{\mathcal{L}},

where :math:`k` is the number of fitted parameters.

The Bayesian information criterion is

.. math::

   \mathrm{BIC}
   =
   k\ln N-2\ln\hat{\mathcal{L}}.

For the four-parameter Gaussian mixture, :math:`k=4`.

The Kolmogorov--Smirnov statistic compares the empirical cumulative
distribution function of the residuals with the fitted mixture,

.. math::

   D_{\mathrm{KS}}
   =
   \sup_{\epsilon}
   \left|
   F_N(\epsilon)
   -
   F_{\hat{\boldsymbol{\phi}}}(\epsilon)
   \right|.

Here :math:`F_N` is the empirical cumulative distribution function,
and :math:`F_{\hat{\boldsymbol{\phi}}}` is the cumulative distribution
function of the fitted mixture.

AIC and BIC are useful for comparing statistical models fitted to
the same catalogue. The KS statistic measures distribution-level
agreement, although standard KS significance thresholds do not
directly apply when the model parameters have been estimated from
the same observations.

The convergence information describes the numerical optimization
and should be inspected separately from statistical goodness of fit.


Assumptions and limitations
---------------------------

The two-component Gaussian mixture provides a compact approximation
to photometric-redshift errors, but it does not capture every feature
of realistic photo-z catalogues.

The main assumptions are:

- The normalized residual distribution is independent of true redshift
  over the fitted sample.
- The core residual mean is fixed to zero.
- The core and outlier components are Gaussian.
- The outlier population is described by a single Gaussian component.
- The catalogue residuals are treated as independent observations.
- The core component satisfies a minimum mixture-weight constraint.
- The calibration catalogue is assumed to represent the population
  to which the fitted model will be applied.

In particular, a single global residual distribution may not adequately
describe samples with strongly redshift-dependent scatter, multiple
outlier populations, or significant selection effects.

The fitted mixture fraction is also model-dependent. It should not
automatically be interpreted as a unique, model-independent
catastrophic-outlier rate.

The method estimates the parameters of the specified statistical model,
rather than reconstructing an arbitrary photo-z error distribution.

For practical examples of parameter recovery, fit diagnostics, and
tomographic reconstruction, see
:doc:`../examples/tomography/photoz_fitting`.
