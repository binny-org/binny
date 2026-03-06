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

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(
               z,
               0.0,
               curve,
               color=color,
               alpha=0.65,
               linewidth=0.0,
               zorder=10 + i,
           )
           ax.plot(
               z,
               curve,
               color="k",
               linewidth=2.2,
               zorder=20 + i,
           )

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)
       ax.set_ylabel(r"$n_i(z)$", fontsize=13)

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

   fig, ax = plt.subplots(figsize=(8.2, 4.8))
   plot_bins(
       ax,
       z,
       specz_result.bins,
       title="Spec-z binning: 4 equal-number bins",
   )
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

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, equidistant_result.bins, "Equidistant")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, equipopulated_result.bins, "Equal-number")

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

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, three_bin_result.bins, "3 bins")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, five_bin_result.bins, "5 bins")

   plt.tight_layout()


Spectroscopic uncertainties
---------------------------

The uncertainty block can also be used in the spectroscopic case, even if
the corresponding values are usually much smaller than for photo-z binning.

Spectroscopic scatter
~~~~~~~~~~~~~~~~~~~~~

This parameter sets the width of the measurement uncertainty in the
spectroscopic redshift estimate. Although spectroscopic redshifts are
typically very precise, a nonzero ``specz_scatter`` broadens the observed
redshift distribution slightly, which smooths the bin edges and produces
a small overlap between neighboring tomographic bins.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, low_scatter_result.bins, "Low spec-z scatter")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, high_scatter_result.bins, "High spec-z scatter")

   plt.tight_layout()


Completeness
~~~~~~~~~~~~

This parameter describes the fraction of galaxies that are successfully
observed and assigned a reliable spectroscopic redshift in each bin.
A value of ``1`` means that all galaxies in the true bin are retained,
while lower values reduce the amplitude of the corresponding tomographic
distribution to reflect incomplete sampling.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, complete_result.bins, "Completeness: all unity")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, reduced_result.bins, "Lower completeness at high $z$")

   plt.tight_layout()


Catastrophic misassignment
~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter sets the fraction of galaxies whose spectroscopic
redshift assignment fails catastrophically. These galaxies are
redistributed into other bins according to the chosen leakage model.
Increasing ``catastrophic_frac`` therefore increases mixing between
otherwise well-separated tomographic bins.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, no_catastrophic_result.bins, "No catastrophic reassignment")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, catastrophic_result.bins, "Increasing catastrophic fraction")

   plt.tight_layout()


Leakage model
~~~~~~~~~~~~~

This parameter defines how galaxies affected by catastrophic redshift
failures are redistributed across bins. The ``neighbor`` model moves
misassigned galaxies primarily into adjacent bins, while ``uniform``
spreads them across all bins with equal probability.

.. plot::
   :include-source: True
   :width: 760

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography

   def plot_bins(ax, z, bin_dict, title, cmap="viridis", cmap_range=(0.0, 1.0)):
       keys = sorted(bin_dict.keys())
       colors = cmr.take_cmap_colors(
           cmap,
           len(keys),
           cmap_range=cmap_range,
           return_fmt="hex",
       )

       for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
           curve = np.asarray(bin_dict[key], dtype=float)
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.65, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

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

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, neighbor_result.bins, "Leakage model: neighbor")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, uniform_result.bins, "Leakage model: uniform")

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
- The plotting style here uses filled tomographic curves with black outlines,
  matching the photometric examples and broader Binny visual style.
- The scheme names used in the API are ``equidistant`` and
  ``equipopulated``.