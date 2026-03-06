Photometric bins
================

This page shows simple, executable examples of how to build
**photometric tomographic redshift bins** from a parent distribution
using :class:`binny.NZTomography`.

In contrast to spectroscopic binning, photometric binning requires a
photo-z uncertainty model. This page therefore introduces not only the
basic photo-z workflow, but also the effect of individual uncertainty
ingredients on the resulting tomographic bins.

The main ideas illustrated are:

- building photo-z bins from a parent :math:`n(z)`,
- comparing binning schemes,
- changing the number of bins,
- varying individual photo-z uncertainty terms,
- and combining all uncertainty ingredients in one setup.

All plotting examples below are executable via ``.. plot::``.


Basic photometric binning
-------------------------

We first construct a simple photo-z tomographic setup using an underlying
Smail distribution, equal-number binning, and a single scalar scatter term.

.. plot::
   :include-source: True
   :width: 640

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 3.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   photoz_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": 0.03,
       },
       "normalize_bins": True,
   }

   photoz_result = tomo.build_bins(z=z, nz=nz, tomo_spec=photoz_spec)

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       len(photoz_result.bins),
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   plt.figure(figsize=(8.2, 4.8))

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       photoz_result.bins.items(),
   ):
       plt.plot(z, bin_curve, lw=2.4, color=color, label=f"Bin {bin_index}")

   plt.xlabel("Redshift $z$", fontsize=14)
   plt.ylabel(r"$n_i(z)$", fontsize=14)
   plt.title("Photo-z binning: 4 equal-number bins", fontsize=14)
   plt.legend(frameon=True, fontsize=11, ncol=2)
   plt.tight_layout()


Changing the binning scheme
---------------------------

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 3.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   common_uncertainties = {"scatter_scale": 0.03}

   equidistant_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equidistant", "n_bins": 4},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   equipop_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   equidistant_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equidistant_spec)
   equipop_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equipop_spec)

   colors = cmr.take_cmap_colors("viridis", 4, cmap_range=(0.15, 0.9), return_fmt="hex")

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (i, curve) in zip(colors, equidistant_result.bins.items()):
       axes[0].plot(z, curve, lw=2.3, color=color)

   axes[0].set_title("Equidistant")
   axes[0].set_xlabel("Redshift $z$")
   axes[0].set_ylabel(r"$n_i(z)$")

   for color, (i, curve) in zip(colors, equipop_result.bins.items()):
       axes[1].plot(z, curve, lw=2.3, color=color)

   axes[1].set_title("Equal-number")
   axes[1].set_xlabel("Redshift $z$")

   plt.tight_layout()


Changing the number of bins
---------------------------

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 3.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   common_uncertainties = {"scatter_scale": 0.03}

   three_bins_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 3},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   five_bins_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 5},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   three = tomo.build_bins(z=z, nz=nz, tomo_spec=three_bins_spec)
   five = tomo.build_bins(z=z, nz=nz, tomo_spec=five_bins_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

   colors3 = cmr.take_cmap_colors("viridis", 3, return_fmt="hex")
   colors5 = cmr.take_cmap_colors("viridis", 5, return_fmt="hex")

   for c, (i, curve) in zip(colors3, three.bins.items()):
       axes[0].plot(z, curve, lw=2.4, color=c)

   axes[0].set_title("3 bins")
   axes[0].set_xlabel("Redshift $z$")
   axes[0].set_ylabel(r"$n_i(z)$")

   for c, (i, curve) in zip(colors5, five.bins.items()):
       axes[1].plot(z, curve, lw=2.2, color=c)

   axes[1].set_title("5 bins")
   axes[1].set_xlabel("Redshift $z$")

   plt.tight_layout()


Photo-z uncertainties
---------------------

Scatter scale
~~~~~~~~~~~~~

.. plot::
   :include-source: True

   import matplotlib.pyplot as plt
   import numpy as np
   import cmasher as cmr

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0, 3, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2,
       beta=1,
       normalize=True,
   )

   low = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {"scatter_scale": 0.03},
       "normalize_bins": True,
   }

   high = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {"scatter_scale": 0.08},
       "normalize_bins": True,
   }

   low_res = tomo.build_bins(z=z, nz=nz, tomo_spec=low)
   high_res = tomo.build_bins(z=z, nz=nz, tomo_spec=high)

   colors = cmr.take_cmap_colors("viridis", 4, return_fmt="hex")

   fig, axes = plt.subplots(1,2,figsize=(11,4.6),sharey=True)

   for c,(i,curve) in zip(colors,low_res.bins.items()):
       axes[0].plot(z,curve,color=c,lw=2.3)

   axes[0].set_title("scatter = 0.03")
   axes[0].set_xlabel("z")
   axes[0].set_ylabel(r"$n_i(z)$")

   for c,(i,curve) in zip(colors,high_res.bins.items()):
       axes[1].plot(z,curve,color=c,lw=2.3)

   axes[1].set_title("scatter = 0.08")
   axes[1].set_xlabel("z")

   plt.tight_layout()


Inspecting the returned bins
----------------------------

.. plot::
   :include-source: True

   import numpy as np
   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0, 3, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2,
       beta=1,
       normalize=True,
   )

   spec = {
       "kind":"photoz",
       "bins":{"scheme":"equipopulated","n_bins":4},
       "uncertainties":{"scatter_scale":0.03},
       "normalize_bins":True,
   }

   result = tomo.build_bins(z=z,nz=nz,tomo_spec=spec)

   print("bin keys:",list(result.bins.keys()))
   print("parent shape:",result.nz.shape)
   print("bin 0 shape:",result.bins[0].shape)
   print("resolved scheme:",result.spec["bins"]["scheme"])


Notes
-----

- These examples use :meth:`binny.NZTomography.build_bins` with a compact
  ``tomo_spec`` mapping.
- Photometric tomography requires an ``uncertainties`` block in addition to
  the parent ``nz`` model and binning setup.
- The uncertainty examples above vary one ingredient at a time so their
  effect on the tomographic curves can be seen more clearly.
- The final combined example follows the full schema style with per-bin
  uncertainty parameters and optional outlier terms.