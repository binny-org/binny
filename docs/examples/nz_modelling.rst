.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Parent n(z) models
=========================

This page provides basic, executable examples showing how to use
:class:`binny.NZTomography` to inspect and evaluate theoretical parent
redshift distributions :math:`n(z)` available through the Binny registry.

These examples focus on **parent distributions**, not tomographic bins.
They are useful for exploring the shapes of built-in models before using
them in a tomography workflow.

All plotting examples below are executable via ``.. plot::``.


Listing available n(z) models
-----------------------------

You can inspect the registry of built-in parent redshift distribution models
through :meth:`binny.NZTomography.list_nz_models`.

.. plot::
   :include-source: True
   :width: 520

   from binny import NZTomography

   models = NZTomography.list_nz_models()

   print(f"Found {len(models)} registered n(z) models:")
   for name in models:
       print(f" - {name}")


Basic Smail model
-----------------

In this example we evaluate a standard Smail-like distribution and plot the
result on a shared redshift grid.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   cmap = "viridis"
   c_smail = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.05, 0.25))[1]

   z = np.linspace(0.0, 3.0, 600)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.28,
       alpha=2.0,
       beta=1.5,
       normalize=True,
   )

   plt.figure(figsize=(7.0, 5.0))
   plt.plot(z, nz, lw=3, color=c_smail)
   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Smail parent redshift distribution", fontsize=15)
   plt.tight_layout()


Gaussian vs gamma
-----------------

Here we compare two common theoretical shapes: a Gaussian distribution and
a gamma-shaped distribution.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   cmap = "viridis"
   c_gauss = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.25, 0.50))[1]
   c_gamma = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.55, 0.80))[1]

   z = np.linspace(0.0, 3.0, 600)

   nz_gaussian = NZTomography.nz_model(
       "gaussian",
       z,
       mu=0.9,
       sigma=0.22,
       normalize=True,
   )

   nz_gamma = NZTomography.nz_model(
       "gamma",
       z,
       k=3.5,
       theta=0.28,
       normalize=True,
   )

   plt.figure(figsize=(7.0, 5.0))
   plt.plot(z, nz_gaussian, lw=3, color=c_gauss, label="Gaussian")
   plt.plot(z, nz_gamma, lw=3, color=c_gamma, label="Gamma")
   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Gaussian vs gamma parent distributions", fontsize=15)
   plt.legend(frameon=True, fontsize=12, loc="upper right")
   plt.tight_layout()


Comparing several registered models
-----------------------------------

In practice, it is often useful to compare several representative theoretical
models on the same redshift grid.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   z = np.linspace(0.0, 3.0, 700)

   cmap = "viridis"
   colors = cmr.take_cmap_colors(
       cmap,
       4,
       cmap_range=(0.08, 0.92),
       return_fmt="hex",
   )
   c1, c2, c3, c4 = colors

   nz_smail = NZTomography.nz_model(
       "smail",
       z,
       z0=0.28,
       alpha=2.0,
       beta=1.5,
       normalize=True,
   )

   nz_gaussian = NZTomography.nz_model(
       "gaussian",
       z,
       mu=0.9,
       sigma=0.22,
       normalize=True,
   )

   nz_gamma = NZTomography.nz_model(
       "gamma",
       z,
       k=3.5,
       theta=0.28,
       normalize=True,
   )

   nz_lognormal = NZTomography.nz_model(
       "lognormal",
       z,
       mu_ln=-0.05,
       sigma_ln=0.45,
       normalize=True,
   )

   plt.figure(figsize=(8.0, 5.4))
   plt.plot(z, nz_smail, lw=3, color=c1, label="Smail")
   plt.plot(z, nz_gaussian, lw=3, color=c2, label="Gaussian")
   plt.plot(z, nz_gamma, lw=3, color=c3, label="Gamma")
   plt.plot(z, nz_lognormal, lw=3, color=c4, label="Lognormal")
   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Example theoretical parent redshift distributions", fontsize=15)
   plt.legend(frameon=True, fontsize=12, loc="upper right")
   plt.tight_layout()


Shifted Smail model
-------------------

A shifted Smail distribution can be useful when you want a low-redshift cutoff
or a delayed onset relative to the standard Smail form.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   cmap = "viridis"
   c_standard = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.10, 0.35))[1]
   c_shifted = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.65, 0.90))[1]

   z = np.linspace(0.0, 3.0, 700)

   nz_standard = NZTomography.nz_model(
       "smail",
       z,
       z0=0.28,
       alpha=2.0,
       beta=1.5,
       normalize=True,
   )

   nz_shifted = NZTomography.nz_model(
       "shifted_smail",
       z,
       z0=0.28,
       alpha=2.0,
       beta=1.5,
       z_shift=0.25,
       normalize=True,
   )

   plt.figure(figsize=(7.0, 5.0))
   plt.plot(z, nz_standard, lw=3, color=c_standard, label="Smail")
   plt.plot(z, nz_shifted, lw=3, color=c_shifted, label="Shifted Smail")
   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Standard vs shifted Smail distribution", fontsize=15)
   plt.legend(frameon=True, fontsize=12, loc="upper right")
   plt.tight_layout()


Compact-support example: tophat
-------------------------------

Some applications benefit from simple compact-support distributions.
Here we show a normalized top-hat model.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   import matplotlib.pyplot as plt
   import cmasher as cmr

   from binny import NZTomography

   cmap = "viridis"
   c_tophat = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.45, 0.70))[1]

   z = np.linspace(0.0, 3.0, 700)

   nz = NZTomography.nz_model(
       "tophat",
       z,
       zmin=0.6,
       zmax=1.2,
       normalize=True,
   )

   plt.figure(figsize=(7.0, 5.0))
   plt.plot(z, nz, lw=3, color=c_tophat)
   plt.xlabel("Redshift $z$", fontsize=15)
   plt.ylabel(r"Normalized $n(z)$", fontsize=15)
   plt.title("Top-hat parent redshift distribution", fontsize=15)
   plt.tight_layout()


Inspecting model values directly
--------------------------------

You can also evaluate a registered model numerically and inspect the returned
array directly before plotting or passing it into a tomography specification.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np

   from binny import NZTomography

   z = np.linspace(0.0, 3.0, 8)

   nz = NZTomography.nz_model(
       "gaussian",
       z,
       mu=1.0,
       sigma=0.25,
       normalize=True,
   )

   print("z grid:")
   print(z)
   print()
   print("n(z) values:")
   print(nz)
   print()
   print("Shape:", nz.shape)
   print("All non-negative:", bool(np.all(nz >= 0.0)))


Notes
-----

- These examples use :meth:`binny.NZTomography.nz_model`, which evaluates
  a registered parent redshift distribution on a supplied redshift grid.
- Setting ``normalize=True`` makes the model integrate to unity over the
  provided grid.
- These are **parent distributions**, not tomographic bins. To construct
  tomographic bins from a parent :math:`n(z)`, see the tomography examples.
- If you add more registered models later, this page can easily be extended
  with additional comparison plots.