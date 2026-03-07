.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Parent n(z) from mocks
=============================

In many realistic workflows, the parent redshift distribution is not chosen
purely by hand. Instead, it is useful to infer an effective analytic model
from a mock or simulated galaxy sample.

Binny provides :meth:`binny.NZTomography.calibrate_smail_from_mock` for this
purpose. Given true galaxy redshifts and apparent magnitudes from a mock
catalog, the calibration estimates:

- the effective Smail shape parameters,
- how the redshift scale parameter varies with limiting magnitude,
- and how the galaxy number density changes with survey depth.

The example below generates a simple synthetic mock catalog, runs the
calibration, and prints the fitted relations.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np

   from binny import NZTomography

   rng = np.random.default_rng(42)

   # ------------------------------------------------------------------
   # Build a simple synthetic mock catalog
   # ------------------------------------------------------------------
   n_gal = 30000

   z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
   z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

   # Make magnitudes loosely correlated with redshift, with scatter
   mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

   maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

   # ------------------------------------------------------------------
   # Run calibration
   # ------------------------------------------------------------------
   result = NZTomography.calibrate_smail_from_mock(
       z_true=z_true,
       mag=mag,
       maglims=maglims,
       area_deg2=5.0,
       infer_alpha_beta_from="deep_cut",
       alpha_beta_maglim=24.5,
       z_max=3.0,
   )

   print("Calibration succeeded:", result["ok"])
   print()

   print("Inferred Smail shape parameters:")
   print(result["alpha_beta_fit"])
   print()

   print("Calibrated z0(maglim) relation:")
   print(result["z0_of_maglim"])
   print()

   print("Calibrated ngal(maglim) relation:")
   print(result["ngal_of_maglim"])


Visualizing the calibrated parent distribution
----------------------------------------------

Once the calibration has been run, you can evaluate the inferred Smail model
on a redshift grid and compare it to the normalized mock redshift histogram.

This is useful for checking whether the fitted analytic parent distribution
provides a reasonable summary of the underlying mock sample.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   rng = np.random.default_rng(42)

   # ------------------------------------------------------------------
   # Synthetic mock catalog
   # ------------------------------------------------------------------
   n_gal = 30000

   z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
   z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

   mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

   maglim = 24.5
   sel = mag <= maglim

   # ------------------------------------------------------------------
   # Calibration
   # ------------------------------------------------------------------
   result = NZTomography.calibrate_smail_from_mock(
       z_true=z_true,
       mag=mag,
       maglims=np.array([22.5, 23.0, 23.5, 24.0, 24.5]),
       area_deg2=5.0,
       infer_alpha_beta_from="deep_cut",
       alpha_beta_maglim=24.5,
       z_max=3.0,
   )

   alpha = result["alpha_beta_fit"]["params"]["alpha"]
   beta = result["alpha_beta_fit"]["params"]["beta"]

   z0_fit = result["z0_of_maglim"]["fit"]
   if z0_fit["law"] == "linear":
       z0 = z0_fit["a"] * maglim + z0_fit["b"]
   elif z0_fit["law"] == "poly2":
       z0 = z0_fit["c2"] * maglim**2 + z0_fit["c1"] * maglim + z0_fit["c0"]
   else:
       raise ValueError(f"Unknown z0 law: {z0_fit['law']}")

   z = np.linspace(0.0, 3.0, 600)
   nz_fit = NZTomography.nz_model(
       "smail",
       z,
       z0=z0,
       alpha=alpha,
       beta=beta,
       normalize=True,
   )

   cmap = "viridis"
   c_hist = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.10, 0.35))[1]
   c_fit = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.65, 0.90))[1]

   plt.figure(figsize=(8.0, 5.2))
   plt.hist(
       z_true[sel],
       bins=50,
       range=(0.0, 3.0),
       density=True,
       alpha=0.45,
       color=c_hist,
       label="Mock sample",
   )
   plt.plot(z, nz_fit, lw=3, color=c_fit, label="Fitted Smail model")

   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Mock redshift sample and calibrated Smail fit", fontsize=15)
   plt.legend(frameon=True, fontsize=12)
   plt.tight_layout()


Inspecting the calibrated depth relations
-----------------------------------------

In addition to reconstructing a parent redshift distribution, the calibration
also returns fitted relations describing how the Smail scale parameter and the
galaxy number density vary with limiting magnitude.

The example below visualizes both sets of calibration points together with
their fitted trends.

.. plot::
   :include-source: True
   :width: 700

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   rng = np.random.default_rng(42)

   n_gal = 30000

   z_true = rng.gamma(shape=2.4, scale=0.32, size=n_gal)
   z_true = z_true[(z_true >= 0.0) & (z_true <= 3.0)]

   mag = 22.0 + 2.2 * z_true + rng.normal(0.0, 0.45, size=z_true.size)

   maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

   result = NZTomography.calibrate_smail_from_mock(
       z_true=z_true,
       mag=mag,
       maglims=maglims,
       area_deg2=5.0,
       infer_alpha_beta_from="deep_cut",
       alpha_beta_maglim=24.5,
       z_max=3.0,
   )

   z0_points = result["z0_of_maglim"]["points"]
   z0_fit = result["z0_of_maglim"]["fit"]

   ngal_points = result["ngal_of_maglim"]["points"]
   ngal_fit = result["ngal_of_maglim"]["fit"]

   mfit = np.linspace(maglims.min(), maglims.max(), 200)

   if z0_fit["law"] == "linear":
       z0_curve = z0_fit["a"] * mfit + z0_fit["b"]
   elif z0_fit["law"] == "poly2":
       z0_curve = z0_fit["c2"] * mfit**2 + z0_fit["c1"] * mfit + z0_fit["c0"]
   else:
       raise ValueError(f"Unknown z0 law: {z0_fit['law']}")

   if ngal_fit["law"] == "linear":
       ngal_curve = ngal_fit["p"] * mfit + ngal_fit["q"]
   elif ngal_fit["law"] == "loglinear":
       ngal_curve = 10.0 ** (ngal_fit["s"] * mfit + ngal_fit["t"])
   else:
       raise ValueError(f"Unknown ngal law: {ngal_fit['law']}")

   cmap = "viridis"
   c1 = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.15, 0.35))[1]
   c2 = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.65, 0.85))[1]

   fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

   axes[0].plot(mfit, z0_curve, lw=3, color=c1)
   axes[0].scatter(z0_points["maglim"], z0_points["z0"], s=45, color=c1)
   axes[0].set_xlabel("Limiting magnitude", fontsize=13)
   axes[0].set_ylabel(r"Fitted $z_0$", fontsize=13)
   axes[0].set_title(r"Calibrated $z_0(m_{\rm lim})$", fontsize=13)

   axes[1].plot(mfit, ngal_curve, lw=3, color=c2)
   axes[1].scatter(ngal_points["maglim"], ngal_points["ngal_arcmin2"], s=45, color=c2)
   axes[1].set_xlabel("Limiting magnitude", fontsize=13)
   axes[1].set_ylabel(r"$n_{\rm gal}$ [arcmin$^{-2}$]", fontsize=13)
   axes[1].set_title(r"Calibrated $n_{\rm gal}(m_{\rm lim})$", fontsize=13)

   plt.tight_layout()


When to use calibration
-----------------------

Calibration is helpful when you have access to a mock catalog or simulation
and want to derive an analytic parent redshift distribution that reflects the
statistical properties of that sample.

This differs from the simpler direct-evaluation examples, where model
parameters are chosen by hand. In practice, both approaches are useful:

- direct evaluation is convenient for quick testing and controlled examples,
- calibration is more appropriate when building survey-motivated parent
  distributions from realistic mock data.