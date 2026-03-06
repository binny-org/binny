Spectroscopic bins
==================

This page shows simple, executable examples of how to build
**spectroscopic tomographic redshift bins** from a parent distribution
using :class:`binny.NZTomography`.

Compared with photometric binning, spectroscopic binning usually assumes
much smaller redshift uncertainties. This makes the tomographic bins
closer to their ideal boundaries, while still allowing controlled
broadening or bin mixing when a spectroscopic uncertainty model is included.

The main ideas illustrated are:

- building spec-z bins from a parent :math:`n(z)`,
- comparing binning schemes,
- changing the number of bins,
- varying spectroscopic uncertainty terms,
- and inspecting the returned result.

All plotting examples below are executable via ``.. plot::``.


Basic spectroscopic binning
---------------------------

We begin with a simple spectroscopic tomographic setup using a Smail parent
distribution, equal-number binning, and a small spectroscopic scatter term.

.. plot::
   :include-source: True
   :width: 640

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   specz_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   specz_result = tomo.build_bins(z=z, nz=nz, tomo_spec=specz_spec)

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       len(specz_result.bins),
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   plt.figure(figsize=(8.2, 4.8))

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       specz_result.bins.items(),
   ):
       plt.plot(z, bin_curve, lw=2.4, color=color, label=f"Bin {bin_index}")

   plt.xlabel("Redshift $z$", fontsize=14)
   plt.ylabel(r"$n_i(z)$", fontsize=14)
   plt.title("Spec-z binning: 4 equal-number bins", fontsize=14)
   plt.legend(frameon=True, fontsize=11, ncol=2)
   plt.tight_layout()


Changing the binning scheme
---------------------------

As in the photometric case, the choice of binning scheme affects where the
tomographic boundaries are placed. Below we compare equidistant and
equal-number spectroscopic bins using the same uncertainty setup.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   common_uncertainties = {
       "completeness": [1.0, 1.0, 1.0, 1.0],
       "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
       "leakage_model": "neighbor",
       "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
   }

   equidistant_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equidistant",
           "n_bins": 4,
       },
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   equipopulated_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   equidistant_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equidistant_spec,
   )
   equipopulated_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=equipopulated_spec,
   )

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       equidistant_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("Equidistant", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       equipopulated_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("Equal-number", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Changing the number of bins
---------------------------

As before, increasing ``n_bins`` gives a finer tomographic partition.
For spectroscopic binning, the boundaries remain sharper because the
uncertainty model is much narrower than in the photo-z case.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   common_three_bin_uncertainties = {
       "completeness": [1.0, 1.0, 1.0],
       "catastrophic_frac": [0.0, 0.0, 0.0],
       "leakage_model": "neighbor",
       "specz_scatter": [0.001, 0.0015, 0.002],
   }

   common_five_bin_uncertainties = {
       "completeness": [1.0, 1.0, 1.0, 1.0, 1.0],
       "catastrophic_frac": [0.0, 0.0, 0.0, 0.0, 0.0],
       "leakage_model": "neighbor",
       "specz_scatter": [0.001, 0.001, 0.0015, 0.0015, 0.002],
   }

   three_bin_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 3,
       },
       "uncertainties": common_three_bin_uncertainties,
       "normalize_bins": True,
   }

   five_bin_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 5,
       },
       "uncertainties": common_five_bin_uncertainties,
       "normalize_bins": True,
   }

   three_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=three_bin_spec)
   five_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=five_bin_spec)

   three_bin_colors = cmr.take_cmap_colors(
       "viridis",
       3,
       cmap_range=(0.2, 0.85),
       return_fmt="hex",
   )
   five_bin_colors = cmr.take_cmap_colors(
       "viridis",
       5,
       cmap_range=(0.1, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       three_bin_colors,
       three_bin_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.4, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("Equal-number: 3 bins", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       five_bin_colors,
       five_bin_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.0, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("Equal-number: 5 bins", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Spectroscopic uncertainties
---------------------------

The uncertainty block can also be used in the spectroscopic case, even if
the corresponding values are usually much smaller than for photo-z binning.

Spectroscopic scatter
~~~~~~~~~~~~~~~~~~~~~

The ``specz_scatter`` parameter controls the width of the measurement
scatter model in observed spectroscopic redshift. Larger values broaden
the bins more strongly.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   low_scatter_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.0005, 0.0005, 0.00075, 0.001],
       },
       "normalize_bins": True,
   }

   high_scatter_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.003, 0.003, 0.004, 0.005],
       },
       "normalize_bins": True,
   }

   low_scatter_result = tomo.build_bins(z=z, nz=nz, tomo_spec=low_scatter_spec)
   high_scatter_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=high_scatter_spec,
   )

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       low_scatter_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("Low spec-z scatter", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       high_scatter_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("High spec-z scatter", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Completeness
~~~~~~~~~~~~

The ``completeness`` term reduces the retained signal in each true bin.
Here we compare a fully complete case with one in which completeness
decreases toward higher-redshift bins.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   complete_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   reduced_completeness_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 0.95, 0.9, 0.8],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   complete_result = tomo.build_bins(z=z, nz=nz, tomo_spec=complete_spec)
   reduced_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=reduced_completeness_spec,
   )

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       complete_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("Completeness: all unity", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       reduced_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("Lower completeness at high $z$", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Catastrophic misassignment
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``catastrophic_frac`` term controls the fraction of galaxies that are
reassigned to other bins according to the chosen leakage model. Increasing
this fraction produces stronger bin mixing.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   no_catastrophic_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.0, 0.0, 0.0],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   catastrophic_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.05, 0.1, 0.15],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   no_catastrophic_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=no_catastrophic_spec,
   )
   catastrophic_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=catastrophic_spec,
   )

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       no_catastrophic_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("No catastrophic reassignment", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       catastrophic_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("Increasing catastrophic fraction", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Leakage model
~~~~~~~~~~~~~

The ``leakage_model`` setting controls how catastrophically reassigned
galaxies are redistributed across observed bins. Here we compare
``neighbor`` and ``uniform`` redistribution.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   neighbor_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.05, 0.1, 0.15],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   uniform_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 1.0, 1.0, 1.0],
           "catastrophic_frac": [0.0, 0.05, 0.1, 0.15],
           "leakage_model": "uniform",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   neighbor_result = tomo.build_bins(z=z, nz=nz, tomo_spec=neighbor_spec)
   uniform_result = tomo.build_bins(z=z, nz=nz, tomo_spec=uniform_spec)

   bin_colors = cmr.take_cmap_colors(
       "viridis",
       4,
       cmap_range=(0.15, 0.9),
       return_fmt="hex",
   )

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       neighbor_result.bins.items(),
   ):
       axes[0].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[0].set_title("Leakage model: neighbor", fontsize=13)
   axes[0].set_xlabel("Redshift $z$", fontsize=13)
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   for color, (bin_index, bin_curve) in zip(
       bin_colors,
       uniform_result.bins.items(),
   ):
       axes[1].plot(z, bin_curve, lw=2.3, color=color, label=f"Bin {bin_index}")

   axes[1].set_title("Leakage model: uniform", fontsize=13)
   axes[1].set_xlabel("Redshift $z$", fontsize=13)

   axes[0].legend(frameon=True, fontsize=10, ncol=2)
   plt.tight_layout()


Inspecting the returned bins directly
-------------------------------------

The object returned by :meth:`binny.NZTomography.build_bins` stores the
parent distribution, the resolved specification, and the bin curves.
You can inspect the contents directly before using the result elsewhere.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np

   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 501)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   specz_spec = {
       "kind": "specz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "completeness": [1.0, 0.95, 0.9, 0.85],
           "catastrophic_frac": [0.0, 0.05, 0.1, 0.15],
           "leakage_model": "neighbor",
           "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
       },
       "normalize_bins": True,
   }

   specz_result = tomo.build_bins(z=z, nz=nz, tomo_spec=specz_spec)

   print("Bin keys:")
   print(list(specz_result.bins.keys()))
   print()
   print("Parent shape:", specz_result.nz.shape)
   print("Bin 0 shape:", specz_result.bins[0].shape)
   print("Resolved kind:", specz_result.spec["kind"])
   print("Resolved scheme:", specz_result.spec["bins"]["scheme"])
   print("Resolved n_bins:", specz_result.spec["bins"]["n_bins"])


Notes
-----

- These examples use :meth:`binny.NZTomography.build_bins` with a compact
  ``tomo_spec`` mapping.
- When building bins from arrays, pass all three pieces explicitly:
  ``z``, ``nz``, and ``tomo_spec``.
- In the spectroscopic case, the supported uncertainty terms follow the
  spec-z schema, such as ``completeness``, ``catastrophic_frac``,
  ``leakage_model``, and ``specz_scatter``.
- The scheme names used in the API are ``equidistant`` and
  ``equipopulated``.