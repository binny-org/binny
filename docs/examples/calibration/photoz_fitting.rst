.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Photometric-redshift error fitting
=========================================

This page shows how to estimate **photometric-redshift uncertainty
parameters** from a catalogue containing true and observed redshifts
using :func:`binny.calibration.fit_photoz_errors`.

Binny fits an unbinned two-component Gaussian mixture to the normalized
photo-z residuals. The fitted parameters can then be converted into
Binny's photometric-redshift uncertainty model and used to construct
tomographic redshift distributions.

The main ideas illustrated are:

- generating a synthetic catalogue with core scatter and outliers,
- fitting and evaluating the photo-z residual distribution,
- comparing injected and recovered uncertainty parameters,
- inspecting optional fitting diagnostics,
- and reconstructing tomographic bins using the fitted model.

All plotting examples below are executable via ``.. plot::``.


Fitting photometric-redshift errors
-----------------------------------

We first generate a synthetic catalogue with a narrow Gaussian core
and a broader, displaced outlier population.

The normalized photo-z residual is defined as

.. math::

   \epsilon =
   \frac{z_{\mathrm{obs}} - z_{\mathrm{true}}}
        {1 + z_{\mathrm{true}}}.

The core component has zero mean and scatter ``scatter_scale``.
The outlier component has its own scatter, mean displacement, and
population fraction.

We fit the catalogue directly, without binning the residuals into
a histogram. The histogram below is used only for visualization.

The :func:`binny.calibration.sample_photoz_errors` helper draws observed
redshifts from the injected model, and
:func:`binny.calibration.photoz_residual_pdf` evaluates the fitted density
and its weighted components directly.

.. plot::
   :include-source: True
   :width: 760

   import matplotlib.pyplot as plt
   import numpy as np

   from binny.style import palette
   from binny.calibration import (
       fit_photoz_errors,
       photoz_residual_pdf,
       sample_photoz_errors,
   )

   rng = np.random.default_rng(42)

   n_galaxies = 100_000

   injected = {
       "scatter_scale": 0.03,
       "outlier_scatter_scale": 0.08,
       "outlier_mean_shift": 0.12,
       "outlier_frac": 0.20,
   }

   z_true = rng.uniform(0.1, 2.0, n_galaxies)

   z_obs = sample_photoz_errors(z_true, **injected, rng=rng)
   residual = (z_obs - z_true) / (1.0 + z_true)

   fit = fit_photoz_errors(
       z_true,
       z_obs,
       n_init=10,
       seed=42,
   )

   x = np.linspace(-0.25, 0.45, 1000)

   pdf, components = photoz_residual_pdf(
       x,
       **fit,
       return_components=True,
   )

   fig, ax = plt.subplots(figsize=(8.6, 4.9))

   ax.hist(
       residual,
       bins=120,
       density=True,
       color=palette[1],
       alpha=0.40,
       edgecolor="none",
       label="Synthetic catalogue",
   )

   ax.plot(
       x,
       pdf,
       color="k",
       lw=2.3,
       label="Fitted mixture",
   )

   ax.plot(
       x,
       components["core"],
       color=palette[0],
       lw=2.0,
       ls="--",
       label="Core",
   )

   ax.plot(
       x,
       components["outlier"],
       color=palette[-1],
       lw=2.0,
       ls="--",
       label="Outliers",
   )

   ax.set_xlabel(r"Normalized photo-z residual $\epsilon$")
   ax.set_ylabel("Probability density")
   ax.set_xlim(-0.25, 0.45)
   ax.legend(frameon=False)

   plt.tight_layout()


Evaluating the residual PDF and CDF
-----------------------------------

The fitted parameter dictionary can be passed directly to both
:func:`binny.calibration.photoz_residual_pdf` and
:func:`binny.calibration.photoz_residual_cdf` without conversion.
The PDF is analytically normalized over the full residual domain, not
renormalized over the supplied grid. With ``return_components=True``,
the returned ``core`` and ``outlier`` densities integrate to
:math:`1-f_{\mathrm{out}}` and :math:`f_{\mathrm{out}}`, respectively.

The CDF gives the probability that a normalized residual is less than
or equal to a specified value. Differences of CDF values give the
probability of falling within a residual interval.

.. code-block:: python

   import numpy as np

   from binny.calibration import (
       photoz_residual_cdf,
       photoz_residual_pdf,
   )

   # Reuse the fitted parameters from the previous example.
   x = np.linspace(-0.25, 0.45, 1000)
   pdf = photoz_residual_pdf(x, **fit)
   cdf = photoz_residual_cdf(x, **fit)

   bounds = np.array([-0.1, 0.1])
   probabilities = photoz_residual_cdf(bounds, **fit)
   interval_probability = probabilities[1] - probabilities[0]

   print("Probability within [-0.1, 0.1]:", interval_probability)

The PDF accepts finite one-dimensional residual arrays, while the CDF
also accepts infinite endpoints. Both require positive finite scatter
scales and an outlier fraction between zero and one.


Comparing injected and fitted parameters
----------------------------------------

The fitting routine returns four parameters describing the core and
outlier populations.

These parameters are estimated simultaneously from the unbinned
residual distribution.

The example below compares the recovered values with the parameters
used to generate the synthetic catalogue.

.. plot::
   :include-source: True
   :width: 760

   import matplotlib.pyplot as plt
   import numpy as np

   from binny.style import palette
   from binny.calibration import (
       fit_photoz_errors,
       sample_photoz_errors,
   )

   rng = np.random.default_rng(42)

   injected = {
       "scatter_scale": 0.03,
       "outlier_scatter_scale": 0.08,
       "outlier_mean_shift": 0.12,
       "outlier_frac": 0.20,
   }

   n = 100_000
   z_true = rng.uniform(0.1, 2.0, n)

   z_obs = sample_photoz_errors(z_true, **injected, rng=rng)

   fit = fit_photoz_errors(z_true, z_obs, seed=42)

   names = list(injected)
   x = np.arange(len(names))

   fig, ax = plt.subplots(figsize=(8.6, 4.9))

   ax.scatter(
       x - 0.12,
       [injected[name] for name in names],
       s=65,
       color="k",
       marker="o",
       label="Injected",
       zorder=3,
   )

   ax.scatter(
       x + 0.12,
       [fit[name] for name in names],
       s=65,
       color=palette[3],
       marker="x",
       label="Fitted",
       zorder=4,
   )

   ax.set_xticks(x)
   ax.set_xticklabels(
       [
           "Core scatter",
           "Outlier scatter",
           "Outlier shift",
           "Outlier fraction",
       ]
   )

   ax.set_ylabel("Parameter value")
   ax.legend(frameon=False)

   plt.tight_layout()


Optional fitting diagnostics
----------------------------

By default, :func:`binny.calibration.fit_photoz_errors`
returns only the four fitted parameters.

Setting ``return_metadata=True`` additionally returns a dictionary
containing numerical and statistical diagnostics.

These include the convergence status, number of EM iterations,
log-likelihood, Akaike information criterion (AIC), Bayesian information
criterion (BIC), and Kolmogorov--Smirnov statistic.

.. code-block:: python

   from binny.calibration import fit_photoz_errors

   # z_true and z_obs are arrays from the synthetic catalogue above.
   fit, metadata = fit_photoz_errors(
       z_true,
       z_obs,
       n_init=10,
       max_iter=1000,
       tol=1e-8,
       seed=42,
       return_metadata=True,
   )

   print("Fitted parameters")
   for name, value in fit.items():
       print(f"{name}: {value:.6f}")

   print("Fit diagnostics")
   print("Converged:", metadata["converged"])
   print("Iterations:", metadata["n_iter"])
   print("Log-likelihood:", metadata["loglike"])
   print("AIC:", metadata["aic"])
   print("BIC:", metadata["bic"])
   print("KS statistic:", metadata["ks_statistic"])

The information criteria are useful for comparing alternative
statistical models fitted to the same catalogue. The KS statistic
measures the discrepancy between the empirical and fitted residual
distributions.

A small KS statistic indicates agreement between the empirical
distribution and the fitted model, but does not by itself establish
that the model is physically adequate.


Converting fitted parameters to Binny uncertainties
---------------------------------------------------

The fitted residual parameters are not identical to the parameters
used internally by Binny's forward photo-z assignment model.

The conversion is handled by
:func:`binny.calibration.to_binny_photoz_params`.

.. code-block:: python

   from binny.calibration import (
       fit_photoz_errors,
       to_binny_photoz_params,
   )

   # Reuse z_true and z_obs from the synthetic catalogue.
   fit = fit_photoz_errors(z_true, z_obs)

   photoz_params = to_binny_photoz_params(fit)

   print(photoz_params)

The returned dictionary contains the core and outlier scatter,
outlier fraction, and the corresponding mean-scale and mean-offset
parameters.

These can be passed directly to the ``uncertainties`` field of a
photometric tomography specification.


Reconstructing tomographic redshift distributions
-------------------------------------------------

We now combine the fitting and binning workflows.

We generate a synthetic catalogue from a Smail parent distribution,
inject a known photo-z uncertainty model, and fit its residuals.

The fitted parameters are then converted into Binny's forward model.
We construct tomographic bins using both the injected and recovered
uncertainties, keeping the parent distribution and bin edges fixed.

This allows us to compare the resulting redshift distributions and
quantify the effect of parameter recovery on tomography.

.. plot::
   :include-source: True
   :width: 760

   import matplotlib.pyplot as plt
   import numpy as np

   from binny.style import palette
   from binny.calibration import (
       fit_photoz_errors,
       sample_photoz_errors,
       to_binny_photoz_params,
   )
   from binny.nz import nz_model, sample_redshifts_from_nz
   from binny.tomography import build_tomography_from_spec

   rng = np.random.default_rng(42)

   z = np.linspace(0.0, 3.0, 1000)

   nz = nz_model(
       "smail",
       z,
       z0=0.5,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   n_galaxies = 100_000

   # Draw true redshifts from the parent distribution.
   z_true = sample_redshifts_from_nz(
       z,
       nz,
       n_galaxies,
       rng=rng,
   )

   injected = {
       "scatter_scale": 0.03,
       "outlier_scatter_scale": 0.08,
       "outlier_mean_shift": 0.12,
       "outlier_frac": 0.20,
   }

   z_obs = sample_photoz_errors(z_true, **injected, rng=rng)

   # Recover the photo-z uncertainty parameters.
   fit = fit_photoz_errors(
       z_true,
       z_obs,
       n_init=10,
       seed=42,
   )

   injected_uncertainties = to_binny_photoz_params(injected)
   fitted_uncertainties = to_binny_photoz_params(fit)

   bin_edges = np.linspace(0.2, 2.2, 6)

   def make_spec(uncertainties):
       return {
           "kind": "photoz",
           "bins": {
               "edges": bin_edges,
           },
           "uncertainties": uncertainties,
           "normalize_bins": True,
       }

   injected_result = build_tomography_from_spec(
       z=z,
       nz=nz,
       tomo_spec=make_spec(injected_uncertainties),
   )

   fitted_result = build_tomography_from_spec(
       z=z,
       nz=nz,
       tomo_spec=make_spec(fitted_uncertainties),
   )

   keys = sorted(injected_result.bins)
   colors = palette[:len(keys)]

   fig, ax = plt.subplots(figsize=(8.6, 4.9))

   for i, (color, key) in enumerate(
       zip(colors, keys, strict=True)
   ):
       injected_curve = np.asarray(
           injected_result.bins[key],
           dtype=float,
       )
       fitted_curve = np.asarray(
           fitted_result.bins[key],
           dtype=float,
       )

       ax.fill_between(
           z,
           0.0,
           injected_curve,
           color=color,
           alpha=0.65,
           linewidth=0.0,
           zorder=10 + i,
       )

       ax.plot(
           z,
           injected_curve,
           color="k",
           lw=1.8,
           zorder=20 + i,
       )

       ax.plot(
           z,
           fitted_curve,
           color=color,
           lw=2.2,
           ls="--",
           zorder=30 + i,
       )

   ax.plot(
       z,
       np.zeros_like(z),
       color="k",
       lw=2.2,
       zorder=1000,
   )

   ax.plot(
       [],
       [],
       color="k",
       lw=1.8,
       label="Injected (filled)",
   )

   ax.plot(
       [],
       [],
       color="k",
       lw=2.2,
       ls="--",
       label="Fitted (dashed)",
   )

   ax.set_title("Injected and fitted photo-z tomography")
   ax.set_xlabel("Redshift $z$")
   ax.set_ylabel(r"Normalized $n_i(z)$")
   ax.set_xlim(0.0, 3.0)
   ax.legend(frameon=False, ncol=2)

   plt.tight_layout()


Quantifying reconstruction accuracy
-----------------------------------

Visual agreement between injected and reconstructed tomographic
distributions is useful, but does not fully characterize the
reconstruction accuracy.

For each tomographic bin, we can compare its population, mean
redshift, and normalized distribution.

The following quantities are particularly useful:

- the relative difference in bin population,
- the shift in mean true redshift,
- and the integrated absolute difference between normalized
  tomographic distributions.

For normalized distributions, the mean redshift is

.. math::

   \bar{z}_i =
   \frac{\int z n_i(z)\,\mathrm{d}z}
        {\int n_i(z)\,\mathrm{d}z}.

The integrated absolute difference is

.. math::

   D_i =
   \int
   \left|
   \hat{n}_i(z) - n_i(z)
   \right|
   \mathrm{d}z,

where the distributions are normalized to unit integral.

The example below evaluates these diagnostics for the reconstructed
bins from the previous section.

.. code-block:: python

   import numpy as np

   # Reuse injected_result, fitted_result, and z from above.
   for key in sorted(injected_result.bins):
       true_bin = np.asarray(injected_result.bins[key])
       fitted_bin = np.asarray(fitted_result.bins[key])

       true_norm = np.trapezoid(true_bin, z)
       fitted_norm = np.trapezoid(fitted_bin, z)

       true_mean = np.trapezoid(z * true_bin, z) / true_norm
       fitted_mean = np.trapezoid(z * fitted_bin, z) / fitted_norm

       true_pdf = true_bin / true_norm
       fitted_pdf = fitted_bin / fitted_norm

       mean_shift = fitted_mean - true_mean
       l1_distance = np.trapezoid(
           np.abs(fitted_pdf - true_pdf),
           z,
       )

       print(
           f"Bin {key + 1}: "
           f"mean-z shift = {mean_shift:+.6f}, "
           f"L1 distance = {l1_distance:.6f}"
       )

Because ``normalize_bins=True`` normalizes each bin separately, the
resulting arrays cannot be used to recover relative bin populations.
Population comparisons require the corresponding unnormalized
tomographic distributions.


Notes
-----

- The fitting routine operates on individual galaxy redshifts rather
  than histogrammed residuals.
- The core Gaussian mean is fixed to zero, while the outlier mean,
  scatter, and fraction are fitted.
- The default minimum core weight is ``0.5``, limiting the fitted
  outlier fraction to at most ``0.5``.
- Multiple initializations help reduce sensitivity to local optima.
- The PDF, CDF, and sampling helpers use the same four residual model
  parameters as the fitting routine.
- Sampling accepts either ``seed`` or ``rng``, but not both. Observed
  redshifts are not clipped or truncated.
- The evaluation and sampling helpers allow outlier fractions from zero
  to one, independently of the minimum core weight used during fitting.
- ``return_metadata=True`` provides optional diagnostics without
  changing the fitted parameter dictionary.
- The conversion helper maps the residual model to Binny's existing
  photometric-redshift uncertainty convention.
- The examples assume a redshift-independent normalized residual
  distribution. More complex catalogues may require additional
  modelling or calibration.
- Agreement in synthetic reconstruction tests does not establish
  accuracy for realistic survey photo-z errors.
