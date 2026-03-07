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
               linewidth=1.8,
               zorder=20 + i,
           )

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)
       ax.set_ylabel(r"$n_i(z)$", fontsize=13)

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 500)

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

   fig, ax = plt.subplots(figsize=(8.2, 4.8))
   plot_bins(
       ax,
       z,
       photoz_result.bins,
       title="Photo-z binning: 4 equipopulated bins",
   )
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

   z = np.linspace(0.0, 2.0, 500)

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

   equipopulated_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": common_uncertainties,
       "normalize_bins": True,
   }

   equidistant_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equidistant_spec)
   equipopulated_result = tomo.build_bins(z=z, nz=nz, tomo_spec=equipopulated_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, equidistant_result.bins, "Equidistant")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, equipopulated_result.bins, "Equal-number")

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

   z = np.linspace(0.0, 2.0, 500)

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

   three_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=three_bins_spec)
   five_bin_result = tomo.build_bins(z=z, nz=nz, tomo_spec=five_bins_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, three_bin_result.bins, "3 bins")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, five_bin_result.bins, "5 bins")

   plt.tight_layout()


Photo-z uncertainties
---------------------

The photo-z model can include a central relation for the mean and scatter,
as well as an optional outlier component with its own fraction, mean relation,
and scatter. The examples below vary one ingredient at a time so that its
effect on the tomographic curves is easier to isolate.


Scatter scale
~~~~~~~~~~~~~

This sets how much the main photometric redshift estimate is blurred around
its expected value. A larger ``scatter_scale`` spreads galaxies more broadly
in observed redshift, so neighboring tomographic bins overlap more strongly
and the bin edges look less sharp.

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
           ax.fill_between(z, 0.0, curve, color=color, alpha=0.6, linewidth=0.0, zorder=10 + i)
           ax.plot(z, curve, color="k", linewidth=2.2, zorder=20 + i)

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.2, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)

   tomo = NZTomography()
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   low_scatter_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {"scatter_scale": 0.03},
       "normalize_bins": True,
   }

   high_scatter_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {"scatter_scale": 0.08},
       "normalize_bins": True,
   }

   low_scatter_result = tomo.build_bins(z=z, nz=nz, tomo_spec=low_scatter_spec)
   high_scatter_result = tomo.build_bins(z=z, nz=nz, tomo_spec=high_scatter_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, low_scatter_result.bins, "scatter_scale = 0.03")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, high_scatter_result.bins, "scatter_scale = 0.08")

   plt.tight_layout()


Mean offset
~~~~~~~~~~~

This adds a systematic shift to the main photo-z relation. A positive
``mean_offset`` moves galaxies toward higher observed redshift than their
true redshift, while a negative value would move them lower.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   zero_offset_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "mean_offset": 0.00,
       },
       "normalize_bins": True,
   }

   shifted_offset_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "mean_offset": 0.15,
       },
       "normalize_bins": True,
   }

   zero_offset_result = tomo.build_bins(z=z, nz=nz, tomo_spec=zero_offset_spec)
   shifted_offset_result = tomo.build_bins(z=z, nz=nz, tomo_spec=shifted_offset_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, zero_offset_result.bins, "mean_offset = 0.00")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, shifted_offset_result.bins, "mean_offset = 0.15")

   plt.tight_layout()


Mean scale
~~~~~~~~~~

This changes the slope of the main mapping between true redshift and observed
photo-z. Values above ``1`` stretch the relation so the separation in observed
redshift grows faster with true redshift, while values below ``1`` compress it.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   unit_scale_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   stretched_scale_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "mean_scale": 1.50,
       },
       "normalize_bins": True,
   }

   unit_scale_result = tomo.build_bins(z=z, nz=nz, tomo_spec=unit_scale_spec)
   stretched_scale_result = tomo.build_bins(z=z, nz=nz, tomo_spec=stretched_scale_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, unit_scale_result.bins, "mean_scale=1.0")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, stretched_scale_result.bins, "mean_scale=1.5")

   plt.tight_layout()


Outlier fraction
~~~~~~~~~~~~~~~~

This gives the fraction of galaxies that do not follow the main photo-z
relation and are instead assigned to a separate outlier component. Increasing
``outlier_frac`` puts more weight into misassigned galaxies, which creates
broader tails and stronger leakage between bins.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   no_outliers_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
       },
       "normalize_bins": True,
   }

   with_outliers_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
           "outlier_mean_scale": 1.50,
       },
       "normalize_bins": True,
   }

   no_outliers_result = tomo.build_bins(z=z, nz=nz, tomo_spec=no_outliers_spec)
   with_outliers_result = tomo.build_bins(z=z, nz=nz, tomo_spec=with_outliers_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, no_outliers_result.bins, "outlier_frac = 0.00")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, with_outliers_result.bins, "outlier_frac = 0.08")

   plt.tight_layout()


Outlier scatter scale
~~~~~~~~~~~~~~~~~~~~~

This sets the width of the outlier component itself. A larger
``outlier_scatter_scale`` makes the outlier population more broadly distributed
in observed redshift, so its contribution is spread over a wider range and
contaminates more bins.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   narrow_outlier_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.15,
           "outlier_mean_offset": 0.06,
           "outlier_mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   broad_outlier_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.35,
           "outlier_mean_offset": 0.06,
           "outlier_mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   narrow_outlier_result = tomo.build_bins(z=z, nz=nz, tomo_spec=narrow_outlier_spec)
   broad_outlier_result = tomo.build_bins(z=z, nz=nz, tomo_spec=broad_outlier_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, narrow_outlier_result.bins, "outlier_scatter_scale = 0.15")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, broad_outlier_result.bins, "outlier_scatter_scale = 0.35")

   plt.tight_layout()


Outlier mean offset
~~~~~~~~~~~~~~~~~~~

TThis shifts the central location of the outlier component relative to the main
relation. Increasing ``outlier_mean_offset`` moves the outlier population
farther away from where the correctly assigned galaxies would lie.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   small_outlier_shift_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.02,
           "outlier_mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   large_outlier_shift_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.2,
           "outlier_mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   small_outlier_shift_result = tomo.build_bins(z=z, nz=nz, tomo_spec=small_outlier_shift_spec)
   large_outlier_shift_result = tomo.build_bins(z=z, nz=nz, tomo_spec=large_outlier_shift_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, small_outlier_shift_result.bins, "outlier_mean_offset = 0.02")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, large_outlier_shift_result.bins, "outlier_mean_offset = 0.12")

   plt.tight_layout()


Outlier mean scale
~~~~~~~~~~~~~~~~~~

This changes how the outlier mean relation grows with true redshift. It
controls whether the outlier population is displaced in a way that stays
roughly proportional to the main relation or becomes increasingly stretched
or compressed with redshift.

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
   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.2,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   unit_outlier_scale_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.2,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
           "outlier_mean_scale": 1.00,
       },
       "normalize_bins": True,
   }

   stretched_outlier_scale_spec = {
       "kind": "photoz",
       "bins": {"scheme": "equipopulated", "n_bins": 4},
       "uncertainties": {
           "scatter_scale": 0.04,
           "outlier_frac": 0.06,
           "outlier_scatter_scale": 0.25,
           "outlier_mean_offset": 0.06,
           "outlier_mean_scale": 1.5,
       },
       "normalize_bins": True,
   }

   unit_outlier_scale_result = tomo.build_bins(z=z, nz=nz, tomo_spec=unit_outlier_scale_spec)
   stretched_outlier_scale_result = tomo.build_bins(z=z, nz=nz, tomo_spec=stretched_outlier_scale_spec)

   fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

   plot_bins(axes[0], z, unit_outlier_scale_result.bins, "outlier_mean_scale = 1.00")
   axes[0].set_ylabel(r"$n_i(z)$", fontsize=13)

   plot_bins(axes[1], z, stretched_outlier_scale_result.bins, "outlier_mean_scale = 1.15")

   plt.tight_layout()


Unified uncertainty model
-------------------------

This combines all photo-z ingredients at once: the width of the main scatter,
any systematic shift in the mean relation, any rescaling of that relation,
and a separate outlier population with its own fraction, offset, slope, and
scatter. In practice this is the most realistic setup, because real
photometric redshift errors usually involve both a main population of roughly
correct estimates and a smaller population of catastrophic failures.

The example below follows the full schema style and uses per-bin values
for all uncertainty ingredients.

.. plot::
   :include-source: True
   :width: 700

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
               linewidth=1.8,
               zorder=20 + i,
           )

       ax.plot(z, np.zeros_like(z), color="k", linewidth=1.8, zorder=1000)
       ax.set_title(title, fontsize=13)
       ax.set_xlabel("Redshift $z$", fontsize=13)
       ax.set_ylabel(r"$n_i(z)$", fontsize=13)

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.5,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   unified_uncertainty_spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": [0.03, 0.04, 0.05, 0.06],
           "mean_offset": [0.00, 0.01, 0.01, 0.02],
           "mean_scale": [1.00, 1.00, 1.00, 1.00],
           "outlier_frac": [0.00, 0.05, 0.1, 0.15],
           "outlier_scatter_scale": [0.00, 0.20, 0.25, 0.30],
           "outlier_mean_offset": [0.00, 0.05, 0.05, 0.08],
           "outlier_mean_scale": [1.00, 1.00, 1.00, 1.00],
       },
       "normalize_bins": True,
   }

   unified_result = tomo.build_bins(
       z=z,
       nz=nz,
       tomo_spec=unified_uncertainty_spec,
   )

   fig, ax = plt.subplots(figsize=(8.6, 4.9))
   plot_bins(
       ax,
       z,
       unified_result.bins,
       title="Unified photo-z uncertainty model",
   )
   plt.tight_layout()


Inspecting the returned bins
----------------------------

.. plot::
   :include-source: True

   import numpy as np
   from binny import NZTomography

   tomo = NZTomography()

   z = np.linspace(0.0, 2.0, 500)

   nz = NZTomography.nz_model(
       "smail",
       z,
       z0=0.5,
       alpha=2.0,
       beta=1.0,
       normalize=True,
   )

   spec = {
       "kind": "photoz",
       "bins": {
           "scheme": "equipopulated",
           "n_bins": 4,
       },
       "uncertainties": {
           "scatter_scale": [0.03, 0.04, 0.05, 0.06],
           "mean_offset": [0.00, 0.01, 0.01, 0.02],
           "mean_scale": [1.00, 1.00, 1.00, 1.00],
           "outlier_frac": [0.00, 0.02, 0.03, 0.05],
           "outlier_scatter_scale": [0.00, 0.20, 0.25, 0.30],
           "outlier_mean_offset": [0.00, 0.05, 0.05, 0.08],
           "outlier_mean_scale": [1.00, 1.00, 1.00, 1.00],
       },
       "normalize_bins": True,
   }

   result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)

   print("bin keys:", list(result.bins.keys()))
   print("parent shape:", result.nz.shape)
   print("bin 0 shape:", result.bins[0].shape)
   print("resolved scheme:", result.spec["bins"]["scheme"])


Notes
-----

- These examples use :meth:`binny.NZTomography.build_bins` with a compact
  ``tomo_spec`` mapping.
- Photometric tomography requires an ``uncertainties`` block in addition to
  the parent ``nz`` model and binning setup.
- The uncertainty examples above vary one ingredient at a time so that the
  role of each parameter can be seen more clearly.
- The unified example follows the full schema style and includes per-bin
  values for the main and outlier components.
- The plotting style here uses filled tomographic curves with black outlines,
  matching the broader Binny visual style more closely than plain line plots.