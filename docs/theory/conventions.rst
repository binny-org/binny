.. |logo| image:: ../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Conventions
==================

This page summarises the notation and conventions used throughout the
Binny documentation and API. The goal is to keep the mathematical
notation used in the documentation consistent with the variable names
used in the code.


Notation
--------

Throughout the documentation we use the following mathematical symbols.

Redshift
^^^^^^^^

- Redshift is denoted by :math:`z`.

- A redshift grid used for evaluating distributions is written as

  .. math::

     z = \{z_0, z_1, \ldots, z_N\}

- All tomographic quantities in Binny are evaluated on a **true redshift
  grid** :math:`z`.


Redshift terminology
^^^^^^^^^^^^^^^^^^^^

Two types of redshift estimates commonly appear in tomographic
analyses.

**Spectroscopic redshift (spec-z)**

- Measured from spectral lines
- Typically very precise
- Often treated as the **true redshift** :math:`z` in analyses

It is commonly abbreviated

- ``specz`` in code
- **spec-z** in text


**Photometric redshift (photo-z)**

- Estimated from broadband photometry
- Subject to measurement uncertainty and scatter
- Often used for defining tomographic bins

Common abbreviations include

- ``photoz`` in code
- ``photo-z`` in text
- ``pz`` in some cosmology literature


True vs observed redshift
^^^^^^^^^^^^^^^^^^^^^^^^^

Tomographic binning often distinguishes between

- **true redshift** :math:`z`
- **observed photometric redshift** :math:`z_{\mathrm{obs}}`

Photometric binning therefore typically involves a **photo-z error model**

.. math::

   p(z_{\mathrm{obs}} \mid z)

which describes the probability of observing
:math:`z_{\mathrm{obs}}` given a galaxy with true redshift :math:`z`.

In Binny, tomographic bin distributions are always returned as
functions of the **true redshift grid** :math:`z`, even when bins are
defined using photometric redshift selections.


Redshift distributions
^^^^^^^^^^^^^^^^^^^^^^

- The redshift distribution of a parent galaxy sample is

  .. math::

     n(z)

- Tomographic bin distributions are written

  .. math::

     n_i(z)

  These represent the contribution of bin :math:`i` to the parent
  distribution :math:`n(z)`, where :math:`i` denotes the bin index.
- When normalized,

  .. math::

     \int n(z)\,\mathrm{d}z = 1

  and

  .. math::

     \int n_i(z)\,\mathrm{d}z

  represents the fraction of galaxies contained in bin :math:`i`.


Bin edges
^^^^^^^^^

Tomographic bins are defined by redshift boundaries

.. math::

   z_i^{\mathrm{min}}, \quad z_i^{\mathrm{max}}

such that the bin window is

.. math::

   W_i(z)


Selection windows
^^^^^^^^^^^^^^^^^

The bin selection function is denoted

.. math::

   W_i(z)

For example, in the case of ideal spectroscopic bins,

.. math::

   W_i(z) =
   \begin{cases}
   1 & z_i^{\mathrm{min}} \le z < z_i^{\mathrm{max}} \\
   0 & \text{otherwise}
   \end{cases}

In photometric tomography this window becomes a **probabilistic
assignment kernel**.


Bin indices
-----------

Tomographic bins are indexed by

.. math::

   i = 0, 1, \ldots, N_{\mathrm{bins}} - 1

This matches the indexing used in Python arrays and dictionaries.

Throughout the documentation:

- :math:`i` denotes a **row index**
- :math:`j` denotes a **column index**


Correlation ordering
--------------------

When correlations between bins are computed, Binny adopts the
convention

.. math::

   i \le j

meaning that only the **upper triangle** of the correlation matrix is
typically used.

This is common in cosmology where correlations such as

.. math::

   C_\ell^{ij}

are symmetric,

.. math::

   C_\ell^{ij} = C_\ell^{ji}

and therefore only the upper triangle needs to be evaluated or stored.

When matrices are visualised, however, the full symmetric matrix may be
shown for clarity.


Sample naming
-------------

In cosmology analyses the terms **lens** and **source** have specific
physical meaning. In Binny we distinguish between two levels of naming.


Generic bin sets
^^^^^^^^^^^^^^^^

For general tomography examples we refer simply to

- **bins**
- **bin sets**
- **tomographic bins**

without assuming a specific cosmological role.


Cosmology-specific naming
^^^^^^^^^^^^^^^^^^^^^^^^^

When used in a large-scale structure context the two most common galaxy
samples are

**Lens sample**

- foreground galaxies used for clustering or galaxy–galaxy lensing

**Source sample**

- background galaxies used for weak lensing measurements


API argument names
------------------

Function arguments in Binny use **plain descriptive names** rather than
mathematical symbols.

For example:

=================== ============================
Argument name       Mathematical meaning
=================== ============================
``z``               :math:`z`
``nz``              :math:`n(z)`
``bin_edges``       :math:`z_i^{\mathrm{min}}, z_i^{\mathrm{max}}`
``bins``            :math:`n_i(z)`
``n_bins``          :math:`N_{\mathrm{bins}}`
=================== ============================

The documentation may use the mathematical notation for clarity,
while the API consistently uses readable argument names.


Data structures
---------------

Tomographic bins returned by Binny typically follow the structure

.. code-block:: python

   bins = {
       0: array([...]),
       1: array([...]),
       2: array([...]),
   }

where

- the dictionary key is the **bin index**
- the value is the **bin distribution evaluated on the redshift grid**


Matrices
--------

Diagnostics that compare bins (overlap matrices, leakage matrices,
correlation matrices) are written mathematically as

.. math::

   M_{ij}

and correspond in Python to nested dictionaries or arrays of the form

.. code-block:: python

   matrix[i][j]


Normalization conventions
-------------------------

Unless otherwise specified:

- parent redshift distributions are normalized

  .. math::

     \int n(z) \mathrm{d}z = 1

- tomographic bins preserve the total number of galaxies

  .. math::

     \sum_i \int n_i(z) \mathrm{d}z = 1


Redshift grids
--------------

All distributions returned by Binny are evaluated on the **same input
redshift grid** provided by the user.

For example

.. code-block:: python

   z = np.linspace(0, 2, 500)

This ensures that

- bins can be compared directly
- diagnostic matrices can be computed consistently
- plotting functions operate on aligned arrays.