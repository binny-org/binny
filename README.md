<p align="center">
  <img src="docs/_static/animations/binny_logo.gif" width="220">
</p>

# binny

<p align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/binny-org/binny/ci.yml?branch=main&label=CI&color=440154&style=flat-square)](https://github.com/binny-org/binny/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/binny-org/binny/docs.yml?branch=main&label=docs&color=31688e&style=flat-square)](https://github.com/binny-org/binny/actions/workflows/docs.yml)
[![License](https://img.shields.io/github/license/binny-org/binny?color=35b779&style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pybinny?color=fde725&style=flat-square)](https://pypi.org/project/pybinny/)
[![Documentation](https://img.shields.io/badge/docs-binny-31688e?style=flat-square)](https://binny-org.github.io/binny)

</p>

**binny** is a Python library for constructing and analyzing  
**tomographic redshift bins** used in cosmology and large-scale structure analyses.

It provides flexible binning algorithms, validation utilities, and diagnostic tools
for forecasting, inference pipelines, and survey analysis workflows.

---

# Installation

## Install from PyPI

```bash
pip install pybinny
````

### Install from source
```bash
git clone https://github.com/binny-org/binny.git
cd binny
python -m pip install -e .
```

### Development install

```bash
python -m pip install -e ".[dev]"
```

---


# Citation

If you use **binny** in your research, please cite it.

```bibtex
@software{sarcevic2026binny,
  title   = {binny: Flexible binning algorithms for cosmology},
  author  = {Šarčević, Nikolina and van der Wild, Matthijs},
  year    = {2026},
  url     = {https://github.com/binny-org/binny}
}
```

Citation metadata is also available in `CITATION.cff`, which GitHub uses to generate citation formats automatically.

---

# Contributing

Contributions are very welcome.
See the **Contributing** guide in the documentation for development workflow,
testing, and code style guidelines.

---

# License

MIT License © 2026 Nikolina Šarčević, Matthijs van der Wild and contributors.

