.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Mock redshift catalog sampling
=====================================

Binny provides utilities for generating individual galaxy redshifts
from a tabulated parent redshift distribution :math:`n(z)`.

This is useful when a workflow requires object-level mock catalogs
rather than only continuous redshift distributions.

The two main functions are:

- :func:`binny.sample_redshifts_from_nz` for drawing continuous redshifts.
- :func:`binny.sample_mock_catalog_from_nz` for generating catalogs with
  optional weights and observed redshifts.

The sampler uses inverse-CDF sampling with exact inversion of the
piecewise-linear distribution between redshift grid nodes.

The examples below demonstrate how to generate mock redshift catalogs,
verify their statistical properties, and apply observational models.


Sampling from a parent n(z)
---------------------------

We begin with a smooth Smail-like parent redshift distribution,

.. math::

   n(z) \propto z^\alpha
   \exp\left[-\left(\frac{z}{z_0}\right)^\beta\right].

The distribution is evaluated on a redshift grid and passed to
``sample_redshifts_from_nz``.

The sampler automatically normalizes the input and returns an array
of continuous redshift samples.

We compare the resulting empirical histogram against the normalized
parent distribution.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_redshifts_from_nz

   z = np.linspace(0.0, 3.0, 600)

   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)
   nz /= np.trapezoid(nz, z)

   z_samples = sample_redshifts_from_nz(
       z=z,
       nz=nz,
       n_samples=100_000,
       seed=42,
   )

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.hist(
       z_samples,
       bins=60,
       range=(0.0, 3.0),
       density=True,
       color=colors[0],
       alpha=0.35,
       edgecolor="none",
       label="Sampled redshifts",
   )

   ax.plot(
       z,
       nz,
       color=colors[1],
       lw=3.0,
       label=r"Input $n(z)$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("Sampling from a parent redshift distribution")
   ax.legend(frameon=False)

   plt.tight_layout()


Effect of catalog size
----------------------

A finite mock catalog is a Monte Carlo realization of the parent
redshift distribution.

Consequently, the empirical histogram fluctuates around the
underlying probability density.

As the number of sampled galaxies increases, these fluctuations
become smaller.

The following example compares catalogs containing different
numbers of galaxies drawn from the same parent distribution.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_redshifts_from_nz

   z = np.linspace(0.0, 3.0, 600)

   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)
   nz /= np.trapezoid(nz, z)

   sample_sizes = [500, 5_000, 100_000]
   z_edges = np.linspace(0.0, 3.0, 51)

   colors = cmr.take_cmap_colors(
       "viridis",
       len(sample_sizes),
       cmap_range=(0.15, 0.85),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   for color, n_samples in zip(colors, sample_sizes, strict=True):
       samples = sample_redshifts_from_nz(
           z,
           nz,
           n_samples=n_samples,
           seed=42,
       )

       hist, _ = np.histogram(
           samples,
           bins=z_edges,
           density=True,
       )

       ax.stairs(
           hist,
           z_edges,
           color=color,
           lw=2.0,
           label=rf"$N={n_samples:,}$",
       )

   ax.plot(
       z,
       nz,
       color="k",
       lw=2.5,
       ls="--",
       label=r"Input $n(z)$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("Convergence with mock catalog size")
   ax.legend(frameon=False)

   plt.tight_layout()


Sampling on a nonuniform redshift grid
--------------------------------------

The redshift sampler does not require equally spaced grid nodes.

This is useful when the input distribution comes from an external
survey product, a simulation, or an adaptive numerical calculation.

The sampling procedure accounts for the width of each redshift
interval when constructing the cumulative probability distribution.

Here we construct a deliberately nonuniform grid and compare the
sampled redshifts against the piecewise-linear input distribution.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_redshifts_from_nz

   u_grid = np.linspace(0.0, 1.0, 35)
   z = 3.0 * u_grid**2

   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)
   nz /= np.trapezoid(nz, z)

   samples = sample_redshifts_from_nz(
       z,
       nz,
       n_samples=100_000,
       seed=42,
   )

   z_fine = np.linspace(0.0, 3.0, 600)
   nz_fine = np.interp(z_fine, z, nz)

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.hist(
       samples,
       bins=60,
       range=(0.0, 3.0),
       density=True,
       color=colors[0],
       alpha=0.35,
       edgecolor="none",
       label="Sampled redshifts",
   )

   ax.plot(
       z_fine,
       nz_fine,
       color=colors[1],
       lw=3.0,
       label="Piecewise-linear input",
   )

   ax.scatter(
       z,
       nz,
       s=25,
       color="k",
       zorder=3,
       label="Input grid nodes",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("Sampling on a nonuniform redshift grid")
   ax.legend(frameon=False)

   plt.tight_layout()


Reproducible random sampling
----------------------------

Mock catalog generation is reproducible when a fixed random seed
is supplied.

Repeated calls with identical inputs and the same seed return
identical redshift samples.

Alternatively, an existing NumPy random generator can be supplied.
This is useful when redshift sampling is part of a larger simulation
requiring a shared random-number stream.

.. doctest::

   >>> import numpy as np
   >>> from binny import sample_redshifts_from_nz
   >>> z = np.linspace(0.0, 3.0, 100)
   >>> nz = z**2 * np.exp(-z / 0.5)
   >>> first = sample_redshifts_from_nz(z, nz, 1000, seed=42)
   >>> second = sample_redshifts_from_nz(z, nz, 1000, seed=42)
   >>> np.array_equal(first, second)
   True
   >>> first.shape
   (1000,)
   >>> bool(np.all((first >= z[0]) & (first <= z[-1])))
   True

The ``seed`` and ``rng`` arguments cannot be supplied simultaneously.


Generating a mock redshift catalog
----------------------------------

The higher-level ``sample_mock_catalog_from_nz`` function generates
a dictionary of NumPy arrays.

By default, the dictionary contains two columns:

``z``
   Sampled redshifts from the input distribution.

``weight``
   Per-object weights, equal to one by default.

The redshifts inherit the interpretation of the input distribution.
For an underlying true-redshift distribution, the sampled values
represent true galaxy redshifts.

.. doctest::

   >>> import numpy as np
   >>> from binny import sample_mock_catalog_from_nz
   >>> z = np.linspace(0.0, 3.0, 300)
   >>> nz = z**2 * np.exp(-(z / 0.55) ** 1.5)
   >>> catalog = sample_mock_catalog_from_nz(
   ...     z=z,
   ...     nz=nz,
   ...     n_samples=10_000,
   ...     seed=42,
   ... )
   >>> sorted(catalog)
   ['weight', 'z']
   >>> catalog["z"].shape
   (10000,)
   >>> np.all(catalog["weight"] == 1.0)
   np.True_

The returned dictionary can be passed to downstream catalog
processing, selection, calibration, or tomography workflows.


Adding object weights
---------------------

The catalog helper optionally accepts a scalar or an array of
per-object weights.

A scalar assigns the same weight to every galaxy, while an array
allows each object to carry a different weight.

The weights do not change how the redshifts are sampled. Instead,
they provide additional information for downstream calculations.

For example, a redshift-dependent selection weight can be applied
after generating a mock catalog.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_mock_catalog_from_nz

   z = np.linspace(0.0, 3.0, 600)

   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)
   nz /= np.trapezoid(nz, z)

   catalog = sample_mock_catalog_from_nz(
       z=z,
       nz=nz,
       n_samples=100_000,
       seed=42,
   )

   z_samples = catalog["z"]

   weights = 1.0 / (1.0 + np.exp(4.0 * (z_samples - 1.2)))

   z_edges = np.linspace(0.0, 3.0, 61)

   unweighted, _ = np.histogram(
       z_samples,
       bins=z_edges,
       density=True,
   )

   weighted, _ = np.histogram(
       z_samples,
       bins=z_edges,
       weights=weights,
       density=True,
   )

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.stairs(
       unweighted,
       z_edges,
       color=colors[0],
       lw=2.5,
       label="Unweighted catalog",
   )

   ax.stairs(
       weighted,
       z_edges,
       color=colors[1],
       lw=2.5,
       label="Selection-weighted catalog",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("Effect of redshift-dependent catalog weights")
   ax.legend(frameon=False)

   plt.tight_layout()

The example applies weights after sampling so that they can depend
on the realized redshifts.

When weights are known in advance, they can instead be passed
directly through the ``weights`` argument of
``sample_mock_catalog_from_nz``.


Adding observed redshifts
-------------------------

The catalog helper supports an optional ``observation_model``
callable.

This allows the underlying sampled redshifts to be transformed
into observed redshifts using a measurement model chosen by
the user.

The callable receives two arguments:

- the sampled redshift array;
- a NumPy random generator.

It must return one finite observed redshift for each sampled
object.

As a simple example, consider a Gaussian measurement model,

.. math::

   z_{\rm obs}
   = z+\sigma_0(1+z)\epsilon,

where

.. math::

   \epsilon\sim\mathcal{N}(0,1).

This is an illustrative observation model, not a separate
built-in Binny photo-z prescription.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_mock_catalog_from_nz

   z = np.linspace(0.0, 3.0, 600)

   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)

   def observation_model(z_samples, rng):
       sigma = 0.05 * (1.0 + z_samples)
       return z_samples + rng.normal(
           loc=0.0,
           scale=sigma,
           size=z_samples.size,
       )

   catalog = sample_mock_catalog_from_nz(
       z=z,
       nz=nz,
       n_samples=100_000,
       observation_model=observation_model,
       seed=42,
   )

   z_true = catalog["z"]
   z_obs = catalog["z_obs"]

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   z_edges = np.linspace(-0.3, 3.3, 75)

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.hist(
       z_true,
       bins=z_edges,
       density=True,
       histtype="step",
       color=colors[0],
       lw=2.5,
       label="True redshifts",
   )

   ax.hist(
       z_obs,
       bins=z_edges,
       density=True,
       histtype="step",
       color=colors[1],
       lw=2.5,
       label="Observed redshifts",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("True and observed redshift distributions")
   ax.legend(frameon=False)

   plt.tight_layout()

The resulting catalog contains ``z``, ``weight``, and ``z_obs``.

The same random generator is used for both parent redshift sampling
and the observation model, ensuring reproducibility.


Visualizing redshift measurement scatter
----------------------------------------

The relationship between sampled and observed redshifts can also
be inspected at the object level.

A scatter plot of observed versus true redshift provides a direct
visualization of the measurement uncertainty.

For the Gaussian example, the scatter increases with redshift
because the measurement standard deviation scales as :math:`1+z`.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import to_rgba

   from binny import sample_mock_catalog_from_nz

   z = np.linspace(0.0, 3.0, 600)
   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)

   def observation_model(z_samples, rng):
       sigma = 0.05 * (1.0 + z_samples)
       return z_samples + rng.normal(
           0.0,
           sigma,
           size=z_samples.size,
       )

   catalog = sample_mock_catalog_from_nz(
       z,
       nz,
       n_samples=10_000,
       observation_model=observation_model,
       seed=42,
   )

   color = cmr.take_cmap_colors(
       "viridis",
       1,
       cmap_range=(0.55, 0.55),
       return_fmt="hex",
   )[0]

   fig, ax = plt.subplots(figsize=(7.5, 5.5))

   ax.scatter(
       catalog["z"],
       catalog["z_obs"],
       s=5,
       color=to_rgba(color, 0.2),
       edgecolors="none",
   )

   ax.plot(
       [0.0, 3.0],
       [0.0, 3.0],
       color="k",
       lw=2.0,
       ls="--",
       label=r"$z_{\rm obs}=z_{\rm true}$",
   )

   ax.set_xlabel(r"True redshift $z_{\rm true}$")
   ax.set_ylabel(r"Observed redshift $z_{\rm obs}$")
   ax.set_title("Object-level redshift measurement scatter")
   ax.legend(frameon=False)

   plt.tight_layout()


Redshift residuals
------------------

For redshift calibration studies, it is often useful to inspect
the residual between observed and true redshifts.

A common normalized residual is

.. math::

   \Delta z
   = \frac{z_{\rm obs}-z_{\rm true}}{1+z_{\rm true}}.

For the illustrative Gaussian model used above,

.. math::

   \Delta z \sim \mathcal{N}(0,\sigma_0^2).

The following example verifies that the normalized residual
distribution follows the expected Gaussian shape.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_mock_catalog_from_nz

   z = np.linspace(0.0, 3.0, 600)
   nz = z**2 * np.exp(-(z / 0.55) ** 1.5)

   sigma0 = 0.05

   def observation_model(z_samples, rng):
       return z_samples + rng.normal(
           0.0,
           sigma0 * (1.0 + z_samples),
           size=z_samples.size,
       )

   catalog = sample_mock_catalog_from_nz(
       z,
       nz,
       n_samples=100_000,
       observation_model=observation_model,
       seed=42,
   )

   residual = (
       (catalog["z_obs"] - catalog["z"])
       / (1.0 + catalog["z"])
   )

   x = np.linspace(-0.25, 0.25, 600)

   gaussian = np.exp(-0.5 * (x / sigma0) ** 2)
   gaussian /= sigma0 * np.sqrt(2.0 * np.pi)

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.hist(
       residual,
       bins=80,
       range=(-0.25, 0.25),
       density=True,
       color=colors[0],
       alpha=0.35,
       edgecolor="none",
       label="Mock residuals",
   )

   ax.plot(
       x,
       gaussian,
       color=colors[1],
       lw=3.0,
       label="Input Gaussian model",
   )

   ax.set_xlabel(
       r"$(z_{\rm obs}-z_{\rm true})/(1+z_{\rm true})$"
   )
   ax.set_ylabel("Probability density")
   ax.set_title("Normalized redshift measurement residuals")
   ax.legend(frameon=False)

   plt.tight_layout()


Sampling from a multimodal distribution
---------------------------------------

The sampler also supports parent redshift distributions that
contain multiple peaks.

This is useful for mock galaxy populations consisting of
multiple redshift components or for testing selection
and calibration methods against nontrivial distributions.

No additional mixture-sampling algorithm is required.
The sampler operates directly on the supplied tabulated curve.

.. plot::
   :include-source: True
   :width: 700

   import cmasher as cmr
   import matplotlib.pyplot as plt
   import numpy as np

   from binny import sample_redshifts_from_nz

   z = np.linspace(0.0, 3.0, 800)

   component_1 = np.exp(
       -0.5 * ((z - 0.7) / 0.15) ** 2
   )

   component_2 = 0.7 * np.exp(
       -0.5 * ((z - 1.8) / 0.25) ** 2
   )

   nz = component_1 + component_2
   nz /= np.trapezoid(nz, z)

   samples = sample_redshifts_from_nz(
       z,
       nz,
       n_samples=100_000,
       seed=42,
   )

   colors = cmr.take_cmap_colors(
       "viridis",
       2,
       cmap_range=(0.2, 0.8),
       return_fmt="hex",
   )

   fig, ax = plt.subplots(figsize=(8.0, 5.2))

   ax.hist(
       samples,
       bins=70,
       range=(0.0, 3.0),
       density=True,
       color=colors[0],
       alpha=0.35,
       edgecolor="none",
       label="Sampled redshifts",
   )

   ax.plot(
       z,
       nz,
       color=colors[1],
       lw=3.0,
       label="Multimodal input",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"Normalized $n(z)$")
   ax.set_title("Sampling from a multimodal redshift distribution")
   ax.legend(frameon=False)

   plt.tight_layout()


Notes
-----

- The input :math:`n(z)` does not need to be normalized.
- Samples are drawn continuously between grid nodes.
- Nonuniform redshift grids are supported.
- The sampled distribution follows the piecewise-linear interpolation
  of the supplied tabulated curve.
- A finite catalog exhibits ordinary Monte Carlo fluctuations.
- Reproducible samples can be generated with ``seed`` or ``rng``.
- Catalog weights do not modify the underlying sampling probabilities.
- Observation models are supplied as callables rather than as a
  separate built-in photo-z prescription.
- The catalog helper does not currently generate outlier flags or
  tomographic bin assignments automatically.
- For mathematical details of the inverse-CDF algorithm, see the
  corresponding theory page on mock redshift catalog sampling.
