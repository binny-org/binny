.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Installation
===================

From PyPI
---------

To install the latest released version of Binny:

.. code-block:: bash

   python -m pip install pybinny

Then import it in Python as:

.. code-block:: python

   import binny

From source
-----------

For development or to use the latest unreleased changes:

.. code-block:: bash

   git clone https://github.com/binny-org/binny.git
   cd binny
   python -m pip install -e .

Development install
-------------------

To install Binny together with development tools for testing, linting, and CI:

.. code-block:: bash

   python -m pip install -e ".[dev]"