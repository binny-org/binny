Development and CI
==================

Continuous Integration
----------------------

Binny uses **GitHub Actions** for continuous integration.
On every push and pull request, the following checks are run:

- ``ruff`` for linting and formatting
- ``pytest`` for unit tests
- Multiple Python versions (3.10, 3.11, 3.12)

The current CI status is shown by the badge on the project README.

Versioning
----------

Binny follows **Semantic Versioning (SemVer)**:

::

   MAJOR.MINOR.PATCH

Version bumps are handled using ``bump-my-version``.

GitHub-based releases
---------------------

Version bumps can be triggered via GitHub Actions:

1. Go to **Actions**
2. Select **Release (bump version)**
3. Click **Run workflow**
4. Choose ``patch``, ``minor``, or ``major``

This will:

- Update ``pyproject.toml``
- Commit the change
- Create a git tag
- Push both the commit and the tag
