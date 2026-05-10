.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Survey presets
=====================

Binny allows survey configurations to be stored as YAML **survey
presets**. A preset groups together the parent redshift distribution,
tomographic binning scheme, uncertainty model, and optional survey
metadata such as footprint or galaxy density.

Using a YAML preset makes it easier to keep survey configurations
consistent across forecasts, examples, and analyses.

LSST survey preset
------------------

The example below shows the LSST survey preset included with Binny.
It defines tomographic configurations for **lens** and **source**
samples for both **year 1** and **year 10**.

.. dropdown:: Show LSST YAML preset
   :icon: code
   :animate: fade-in-slide-down

   .. literalinclude:: ../../src/binny/surveys/configs/lsst_survey_specs.yaml
      :language: yaml
      :caption: LSST survey configuration


Visualizing the preset
~~~~~~~~~~~~~~~~~~~~~~

The figure below loads the LSST preset directly through Binny, constructs
the corresponding tomographic bins, and plots the resulting redshift
distributions for lens and source samples in years 1 and 10.

.. plot::
   :include-source: False
   :width: 900

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography


   def plot_bins(ax, result, title):
       z = result.z
       bin_dict = result.bins
       keys = sorted(bin_dict.keys())

       colors = cmr.take_cmap_colors(
           "viridis",
           len(keys),
           cmap_range=(0.1, 0.9),
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")


   results = {}

   for role in ["lens", "source"]:
       for year in ["1", "10"]:
           tomo = NZTomography()
           results[(role, year)] = tomo.build_survey_bins(
               "lsst",
               role=role,
               year=year,
               include_tomo_metadata=True,
           )


   fig, axes = plt.subplots(
       2,
       2,
       figsize=(11.5, 8.0),
   )

   panel_order = [
       (("lens", "1"), "Lens bins Y1"),
       (("source", "1"), "Source bins Y1"),
       (("lens", "10"), "Lens bins Y10"),
       (("source", "10"), "Source bins Y10"),
   ]

   for ax, (key, title) in zip(axes.ravel(), panel_order, strict=True):
       plot_bins(ax, results[key], title)

       role, year = key
       if role == "lens":
           ax.set_xlim(0.0, 1.5)

   axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
   axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")

   plt.suptitle("LSST survey preset tomography", fontsize=16)

   plt.tight_layout(rect=(0, 0, 1, 0.97))


Euclid survey preset
--------------------

Binny can also be used with a Euclid-inspired survey preset containing
a photometric **source** sample and a spectroscopic **lens** sample.

In this simplified preset, the source sample follows the commonly used
Euclid weak-lensing redshift distribution with 10 explicit tomographic
bin edges, while the lens sample is represented as a spectroscopic
sample over the Euclid clustering redshift range.

.. dropdown:: Show Euclid YAML preset
   :icon: code
   :animate: fade-in-slide-down

   .. literalinclude:: ../../src/binny/surveys/configs/euclid_survey_specs.yaml
      :language: yaml
      :caption: Euclid survey configuration


Visualizing the preset
~~~~~~~~~~~~~~~~~~~~~~

The figure below loads the Euclid preset directly through Binny and
visualizes the resulting spectroscopic lens bins and photometric source
bins.

.. plot::
   :include-source: False
   :width: 900

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography


   def plot_bins(ax, result, title):
       z = result.z
       bin_dict = result.bins
       keys = sorted(bin_dict.keys())

       colors = cmr.take_cmap_colors(
           "viridis",
           len(keys),
           cmap_range=(0.1, 0.9),
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")


   results = {}

   for role in ["lens", "source"]:
       tomo = NZTomography()
       results[role] = tomo.build_survey_bins(
           "euclid",
           role=role,
           year="nominal",
           include_tomo_metadata=True,
       )


   fig, axes = plt.subplots(
       1,
       2,
       figsize=(11.5, 4.8),
   )

   panel_order = [
       ("lens", "Euclid lens bins"),
       ("source", "Euclid source bins"),
   ]

   for ax, (role, title) in zip(axes, panel_order, strict=True):
       plot_bins(ax, results[role], title)

   axes[0].set_xlim(0.75, 1.9)
   axes[1].set_xlim(0.0, 2.6)

   axes[0].set_ylabel(r"Normalized $n_i(z)$")

   plt.tight_layout()


DES survey preset
-----------------

Binny also includes a simplified configuration inspired by the
tomographic setup used in the Dark Energy Survey (DES).

Compared to LSST, DES covers a smaller area of the sky and has lower
galaxy number densities, resulting in fewer tomographic bins and a
shallower redshift reach.

.. dropdown:: Show DES YAML preset
   :icon: code
   :animate: fade-in-slide-down

   .. literalinclude:: ../../src/binny/surveys/configs/des_survey_specs.yaml
      :language: yaml
      :caption: DES survey configuration


Visualizing the preset
~~~~~~~~~~~~~~~~~~~~~~

The figure below loads the DES preset directly through Binny and
visualizes the resulting lens and source tomographic bins.

.. plot::
   :include-source: False
   :width: 850

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography


   def plot_bins(ax, result, title):
       z = result.z
       bin_dict = result.bins
       keys = sorted(bin_dict.keys())

       colors = cmr.take_cmap_colors(
           "viridis",
           len(keys),
           cmap_range=(0.1, 0.9),
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0)

       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")


   results = {}

   for role in ["lens", "source"]:
       tomo = NZTomography()
       results[role] = tomo.build_survey_bins(
           "des",
           role=role,
           year="y1",
           include_tomo_metadata=True,
       )


   fig, axes = plt.subplots(
       1,
       2,
       figsize=(10.5, 4.8),
   )

   panel_order = [
       ("lens", "DES lens bins"),
       ("source", "DES source bins"),
   ]

   for ax, (role, title) in zip(axes, panel_order, strict=True):
       plot_bins(ax, results[role], title)

       if role == "lens":
           ax.set_xlim(0.0, 1.0)

   axes[0].set_ylabel(r"Normalized $n_i(z)$")

   plt.tight_layout()


Roman survey preset
-------------------

Binny also includes a Roman-inspired survey preset based on the
Wenzl22 HLS and wide survey configurations.

The preset contains photometric **lens** and **source** samples for
the optimistic HLS, conservative HLS, and wide Roman scenarios.

.. dropdown:: Show Roman YAML preset
   :icon: code
   :animate: fade-in-slide-down

   .. literalinclude:: ../../src/binny/surveys/configs/roman_survey_specs.yaml
      :language: yaml
      :caption: Roman survey configuration


Visualizing the preset
~~~~~~~~~~~~~~~~~~~~~~

The figure below loads the Roman preset directly through Binny and
visualizes the resulting photometric lens and source tomographic bins.
The top row compares the optimistic High Latitude Survey (HLS)
configuration against the conservative HLS configuration, while the
bottom row shows the wide survey configuration.

.. plot::
   :include-source: False
   :width: 900

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography


   def get_bin_colors(bin_dict):
       keys = sorted(bin_dict.keys())

       colors = cmr.take_cmap_colors(
           "viridis",
           len(keys),
           cmap_range=(0.1, 0.9),
           return_fmt="hex",
       )

       return keys, colors


   def plot_bins(ax, result, title):
       z = result.z
       bin_dict = result.bins
       keys, colors = get_bin_colors(bin_dict)

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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")


   def plot_bins_dashed(ax, result):
       z = result.z
       bin_dict = result.bins
       keys, colors = get_bin_colors(bin_dict)

       for i, key in enumerate(keys):
           curve = np.asarray(bin_dict[key], dtype=float)

           ax.plot(
               z,
               curve,
               color="k",
               linewidth=1.8,
               linestyle="--",
               zorder=120 + i,
           )


   results = {}

   for scenario in ["hls_optimistic", "hls_conservative", "wide"]:
       for role in ["lens", "source"]:
           tomo = NZTomography()
           results[(role, scenario)] = tomo.build_survey_bins(
               "roman",
               role=role,
               scenario=scenario,
               include_tomo_metadata=True,
           )


   fig, axes = plt.subplots(
       2,
       2,
       figsize=(11.5, 8.0),
   )

   plot_bins(
       axes[0, 0],
       results[("lens", "hls_optimistic")],
       "Roman HLS lens bins",
   )
   plot_bins_dashed(
       axes[0, 0],
       results[("lens", "hls_conservative")],
   )

   plot_bins(
       axes[0, 1],
       results[("source", "hls_optimistic")],
       "Roman HLS source bins",
   )
   plot_bins_dashed(
       axes[0, 1],
       results[("source", "hls_conservative")],
   )

   plot_bins(
       axes[1, 0],
       results[("lens", "wide")],
       "Roman wide lens bins",
   )

   plot_bins(
       axes[1, 1],
       results[("source", "wide")],
       "Roman wide source bins",
   )

   axes[0, 0].set_xlim(0.0, 4.0)
   axes[0, 1].set_xlim(0.0, 4.0)
   axes[1, 0].set_xlim(0.0, 4.0)
   axes[1, 1].set_xlim(0.0, 4.0)

   axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
   axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")

   axes[0, 0].plot([], [], color="k", linewidth=1.8, label="HLS optimistic")
   axes[0, 0].plot(
       [],
       [],
       color="k",
       linewidth=1.8,
       linestyle="--",
       label="HLS conservative",
   )
   axes[0, 0].legend(frameon=False)

   plt.suptitle("Roman survey preset tomography", fontsize=16)

   plt.tight_layout(rect=(0, 0, 1, 0.97))


DESI survey preset
------------------

Binny also includes a DESI-inspired survey preset containing
tabulated spectroscopic **LRG** and **ELG** lens redshift
distributions.

The default preset uses the legacy Dani redshift windows:
:math:`0.4 \leq z \leq 1.0` for LRGs and
:math:`0.6 \leq z \leq 1.5` for ELGs.

.. dropdown:: Show DESI YAML preset
   :icon: code
   :animate: fade-in-slide-down

   .. literalinclude:: ../../src/binny/surveys/configs/desi_survey_specs.yaml
      :language: yaml
      :caption: DESI survey configuration


Visualizing the preset
~~~~~~~~~~~~~~~~~~~~~~

The figure below loads the DESI preset directly through Binny and
visualizes the tabulated LRG and ELG redshift distributions. The rows
show the same redshift windows split into one, three, and five
equal-width tomographic bins.

.. plot::
   :include-source: False
   :width: 900

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import NZTomography


   def plot_bins(ax, result, title):
       z = result.z
       bin_dict = result.bins
       keys = sorted(bin_dict.keys())

       colors = cmr.take_cmap_colors(
           "viridis",
           len(keys),
           cmap_range=(0.1, 0.9),
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

       ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

       ax.set_title(title)
       ax.set_xlabel("Redshift $z$")


   edges = {
       ("lrg", "one_bin"): [0.4, 1.0],
       ("lrg", "three_bins"): [0.4, 0.6, 0.8, 1.0],
       ("lrg", "five_bins"): [0.4, 0.52, 0.64, 0.76, 0.88, 1.0],
       ("elg", "one_bin"): [0.6, 1.5],
       ("elg", "three_bins"): [0.6, 0.9, 1.2, 1.5],
       ("elg", "five_bins"): [0.6, 0.78, 0.96, 1.14, 1.32, 1.5],
   }


   results = {}

   for key, bin_edges in edges.items():
       sample, _ = key

       tomo = NZTomography()
       results[key] = tomo.build_survey_bins(
           "desi",
           role="lens",
           sample=sample,
           overrides={"bins": {"edges": bin_edges}},
           include_tomo_metadata=True,
       )


   fig, axes = plt.subplots(
       3,
       2,
       figsize=(11.5, 11.0),
   )

   panel_order = [
       (("lrg", "one_bin"), "DESI LRG: one bin"),
       (("elg", "one_bin"), "DESI ELG: one bin"),
       (("lrg", "three_bins"), "DESI LRG: three bins"),
       (("elg", "three_bins"), "DESI ELG: three bins"),
       (("lrg", "five_bins"), "DESI LRG: five bins"),
       (("elg", "five_bins"), "DESI ELG: five bins"),
   ]

   for ax, (key, title) in zip(axes.ravel(), panel_order, strict=True):
       plot_bins(ax, results[key], title)

   axes[0, 0].set_xlim(0.35, 1.05)
   axes[1, 0].set_xlim(0.35, 1.05)
   axes[2, 0].set_xlim(0.35, 1.05)

   axes[0, 1].set_xlim(0.55, 1.55)
   axes[1, 1].set_xlim(0.55, 1.55)
   axes[2, 1].set_xlim(0.55, 1.55)

   axes[0, 0].set_ylabel(r"Normalized $n_i(z)$")
   axes[1, 0].set_ylabel(r"Normalized $n_i(z)$")
   axes[2, 0].set_ylabel(r"Normalized $n_i(z)$")

   plt.suptitle("DESI survey preset tomography", fontsize=16)

   plt.tight_layout(rect=(0, 0, 1, 0.97))
