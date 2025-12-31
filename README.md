# binny

[![CI](https://github.com/binny-org/binny/actions/workflows/ci.yml/badge.svg)](https://github.com/binny-org/binny/actions/workflows/ci.yml)

**binny** is a Python library providing flexible and well-tested
tomographic binning algorithms for cosmology and related scientific workflows.

It is designed to be simple, explicit, and easy to integrate into forecasting,
inference, and data-processing pipelines.

---

## Features

- Multiple binning schemes:
  - Equidistant (linear)
  - Log-spaced
  - Equal-number / equipopulated
  - Mixed and segmented binning strategies
- Validation utilities for bin edges, bin counts, and axis inputs
- NumPy/SciPy-based, with minimal dependencies
- Fully unit-tested with CI on multiple Python versions

---

## Installation

### From source (recommended during development)

```bash
git clone https://github.com/binny-org/binny.git
cd binny
python -m pip install -e .
````

### Development install (with linting and tests)

```bash
python -m pip install -e ".[dev]"
```

---

## Quickstart

Once installed, you can import the package:

```python
import binny
```

As the public API stabilizes, specific functions and classes will be documented
and exposed here.

Run the test suite:

```bash
pytest
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

---

## Continuous Integration (CI)

This project uses **GitHub Actions** for continuous integration.
On every push and pull request, the following are run:

* `ruff` (linting and formatting checks)
* `pytest` (unit tests)
* Multiple Python versions (3.10, 3.11, 3.12)

CI status is shown in the badge at the top of this README.

---

## Versioning

**binny** follows **Semantic Versioning (SemVer)**:

```
MAJOR.MINOR.PATCH
```

Version bumps are handled automatically using `bump-my-version`.

### Manual version bump via GitHub Actions

1. Go to **Actions**
2. Select **Release (bump version)**
3. Click **Run workflow**
4. Choose `patch`, `minor`, or `major`

This will:

* Update `pyproject.toml`
* Commit the change
* Create a git tag
* Push both the commit and tag to the repository

### Local version bump (optional)

```bash
python -m bumpversion patch
# or: minor / major
```

---

## Citation

If you use **binny** in academic work, please cite it.
Citation metadata is provided in `CITATION.cff`, which GitHub uses to generate
BibTeX and other citation formats automatically.

---

## Project Status

This project is under active development.
The API may evolve as additional binning strategies and utilities are added.

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## Acknowledgements

This package was developed for use in cosmology and large-scale structure
analyses, with an emphasis on clarity, reproducibility, and robust testing.

````

---

