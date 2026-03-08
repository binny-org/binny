.. |logo| image:: _static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Contributing
===================

Contributions to **binny** are very welcome.

This project aims to provide clear, reliable, and well-tested tomographic
binning utilities for cosmology and related scientific workflows.
Contributions that improve correctness, clarity, documentation, or usability
are all valued.


Getting started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your-username>/binny.git
      cd binny

3. Install in editable mode with development dependencies:

   .. code-block:: bash

      python -m pip install -e ".[dev]"


Development workflow
--------------------

Binny uses a standard pull-request-based workflow:

- Create a feature branch from ``main``
- Make focused, logically grouped commits
- Open a pull request against ``main``

Please keep pull requests reasonably scoped. Large or conceptual changes are
best discussed in an issue first.


Code style
----------

Binny follows a consistent and explicit coding style:

- Code is formatted and linted using **ruff**
- The target Python version is **Python ≥ 3.10**
- Type hints are encouraged where they improve clarity

Before opening a pull request, please run:

.. code-block:: bash

   ruff check .
   ruff format --check .


Testing
-------

All new functionality should be accompanied by appropriate unit tests.

Binny uses **pytest** for testing. To run the full test suite:

.. code-block:: bash

   pytest

Tests should:

- Be deterministic
- Avoid unnecessary I/O
- Clearly document expected behavior


Documentation
-------------

Documentation is built using **Sphinx**.

If your contribution affects the public API or user-facing behavior, please
update or add relevant documentation under ``docs/``.

To build the documentation locally:

.. code-block:: bash

   tox -e do

This will:

- Generate API reference pages
- Run doctests
- Build the HTML documentation


Continuous integration
----------------------

Binny uses **GitHub Actions** for continuous integration.

On every push and pull request, the following checks are run:

- ``ruff`` for linting and formatting
- ``pytest`` for unit tests
- Multiple Python versions (3.10, 3.11, 3.12)

Pull requests must pass all CI checks before being merged.

The current CI status is shown by the badge on the project README.


Versioning and releases
-----------------------

Binny follows **Semantic Versioning (SemVer)**:

::

   MAJOR.MINOR.PATCH

Version bumps are handled using ``bump-my-version``.

Releases can be triggered via GitHub Actions:

1. Go to **Actions**
2. Select **Release (bump version)**
3. Click **Run workflow**
4. Choose ``patch``, ``minor``, or ``major``

This workflow will:

- Update ``pyproject.toml``
- Commit the version change
- Create a git tag
- Push both the commit and the tag

Contributors do not need to manually update version numbers unless explicitly
requested.


Questions and discussion
------------------------

If you are unsure about a proposed change, or want feedback before implementing
it, feel free to open an issue or start a discussion on GitHub.

Thoughtful questions and design discussions are always welcome.