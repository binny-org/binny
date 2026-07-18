Predefined bins
===============

Binny can load tomographic redshift distributions that were constructed
externally. This is useful when the bin curves come from a measurement,
simulation, collaboration data product, or another analysis pipeline.

Unlike photo-z and spec-z tomography, predefined bins are not passed through
an assignment model and are not re-binned. The supplied curves are validated
and returned directly.

Loading predefined bins from a file
-----------------------------------

Suppose a text file contains a shared redshift grid followed by one column for
each tomographic bin:

.. code-block:: text

   z bin_0 bin_1 bin_2
   0.00 0.0000 0.0000 0.0000
   0.01 0.0012 0.0000 0.0000
   0.02 0.0048 0.0001 0.0000
   ...
   2.99 0.0000 0.0001 0.0020
   3.00 0.0000 0.0000 0.0000

The corresponding survey configuration can define a predefined tomography
entry:

.. code-block:: yaml

   name: predefined_example

   tomography:
     - role: source
       sample: external_source_sample
       name: externally supplied source bins
       kind: predefined

       bins:
         scheme: predefined

         source:
           path: predefined_bins.txt
           z_col: z
           bin_cols:
             0: bin_0
             1: bin_1
             2: bin_2

         normalize_bins: false
         norm_method: trapezoid

Relative file paths are resolved from Binny's packaged survey-data directory.
An absolute path may also be supplied.

The bins can then be loaded through the standard tomography API:

.. code-block:: python

   from binny import NZTomography

   tomography = NZTomography()

   result = tomography.build_bins(
       config_file="predefined_example.yaml",
       role="source",
       sample="external_source_sample",
       include_tomo_metadata=True,
   )

   print(result.bin_keys)
   print(result.z.shape)

   for bin_index, curve in result.bins.items():
       print(bin_index, curve.shape)

Inspecting the result
---------------------

The returned object provides the same convenience methods as photo-z and
spec-z tomography results:

.. code-block:: python

   shape_statistics = result.shape_stats()
   population_statistics = result.population_stats()

   print(shape_statistics["per_bin"])
   print(population_statistics["fractions"])

Population statistics require tomography metadata, so
``include_tomo_metadata=True`` must be used when building the bins.

Normalizing the supplied bins
-----------------------------

Set ``normalize_bins: true`` to normalize every returned bin independently:

.. code-block:: yaml

   bins:
     scheme: predefined

     source:
       path: predefined_bins.txt
       z_col: z
       bin_cols:
         0: bin_0
         1: bin_1
         2: bin_2

     normalize_bins: true
     norm_method: simpson

With this option, each returned curve satisfies

.. math::

   \int n_i(z)\,\mathrm{d}z = 1.

Building directly from arrays
-----------------------------

Predefined tomography can also be constructed directly from arrays without a
configuration file:

.. code-block:: python

   import numpy as np

   from binny import NZTomography

   z = np.linspace(0.0, 2.0, 201)

   bins = {
       0: 1.4 * np.exp(-0.5 * ((z - 0.45) / 0.18) ** 2),
       1: 0.9 * np.exp(-0.5 * ((z - 1.05) / 0.24) ** 2),
       2: 0.6 * np.exp(-0.5 * ((z - 1.60) / 0.20) ** 2),
   }

   tomography = NZTomography()

   result = tomography.build_predefined_bins(
       z=z,
       bins=bins,
       normalize_bins=False,
       include_tomo_metadata=True,
   )

   print(result.bin_keys)

   for bin_index, curve in result.bins.items():
       print(bin_index, curve.shape)

File format requirements
------------------------

The input file must satisfy the following requirements:

* The redshift grid must be one-dimensional and strictly increasing.
* Every bin curve must have the same shape as the redshift grid.
* Bin values must be finite and non-negative.
* Bin indices must be integers.
* Each bin must have a positive integrated population.
* Named and positional column selectors must not be mixed.

For a headerless table, integer column positions may be used instead:

.. code-block:: yaml

   source:
     path: predefined_bins.txt
     z_col: 0
     bin_cols:
       0: 1
       1: 2
       2: 3

The optional ``n_bins`` field can be used to validate the number of loaded
columns:

.. code-block:: yaml

   bins:
     scheme: predefined
     n_bins: 3

     source:
       path: predefined_bins.txt
       z_col: z
       bin_cols:
         0: bin_0
         1: bin_1
         2: bin_2

Binny raises an error if ``n_bins`` does not match the number of bin columns
loaded from the file.
