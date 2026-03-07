binny
=====

.. image:: _static/assets/logo.png
   :alt: Binny logo
   :width: 150px
   :align: right

**Binny** is a Python library providing flexible, explicit, and well-tested
tomographic binning algorithms for cosmology and related scientific workflows.

It is designed to integrate easily into forecasting, inference, and
data-processing pipelines, with an emphasis on clarity, reproducibility,
and robust validation.

Overview
--------

Binny provides tools for constructing and validating tomographic binning
schemes commonly used in cosmology and large-scale structure analyses.

The package focuses on explicit bin definitions, reproducible binning
strategies, and diagnostics that help compare different tomographic choices.

Typical use cases include:

- Photometric-redshift tomography
- Spectroscopic-redshift binning
- Forecasting and sensitivity studies
- Survey-specific bin definitions

For the mathematical definitions and theoretical background underlying these
binning strategies, see :doc:`theory/index`.

For practical examples of how to use binny in realistic workflows,
see :doc:`examples/index`.


.. toctree::
   :maxdepth: 1
   :caption: Contents

   theory/index
   workflow
   examples/index
   installation
   citation
   development
   contributing
   api/index