# Changelog

## 1.3.0 - 2026-11-13

### Added

- Mock redshift catalog sampling from tabulated parent distributions using exact inverse-CDF sampling.
- Support for nonuniform redshift grids, reproducible sampling, optional catalog weights, and user-defined observation models.
- Unit tests, theoretical documentation, and executable examples.

## 1.2.0 - 2026-11-12

### Changed

- Automated PyPI publishing and GitHub release creation through GitHub Actions.
- Updated Ruff and pre-commit configuration.
- Reformatted benchmark and documentation scripts.
- Updated release metadata and Sphinx configuration.

## *1.1.0 - 2026-11-12*

### *Added*

- Generalized \(\alpha\)–\(\beta\) tomographic binning schemes.
- Flexible coordinate grids and sampling strategies.
- Support for explicit bin edges, custom bin ranges, and mixed binning schemes.
- Unified binning specifications and scheme registry.
- Support for generalized binning in survey YAML configurations.
- New theoretical documentation, examples, and visualizations.

### *Changed*

- Unified bin-edge resolution across photo-z and spec-z tomography.
- Refactored mixed-segment binning and coordinate handling.
- Extended the public API with `build_tomography_from_spec` and additional binning utilities.
- Expanded tests and updated documentation for the generalized binning framework.

### *Compatibility*

- Updated binning interfaces and configuration conventions; existing code using previous interfaces may require migration.

## 1.0.0 - 2026-11-10

### Breaking changes

- Replaced the legacy stateful tomography API with a functional API.
- Removed the `NZTomography` and `TomographyBins` classes.
- Removed backward compatibility with the previous tomography interfaces.
- Updated tomography construction, binning, composition, diagnostics, and survey workflows to use the new API.

### Added

- Introduced the `Tomography` data container and `build_tomography()` function.
- Added `build_survey_tomography()` for constructing tomography from survey presets.
- Added `build_predefined_tomography()` and `load_predefined_bins()` for externally supplied tomographic distributions.
- Added support for custom assignment functions in the generic binning engine.
- Added reusable bin construction with shared normalization, validation, and metadata handling.
- Added functional utilities for combining, comparing, and analyzing tomographic distributions.
- Added photo-z core-and-tail error fitting utilities.
- Added support for combining parent redshift distributions defined on different grids through interpolation.

### Changed

- Reorganized the package into dedicated modules for axes, binning, tomography, redshift-distribution modelling, surveys, and correlations.
- Consolidated common binning, normalization, and validation functionality.
- Simplified tomography construction and reduced reliance on mutable internal state.
- Updated tests, examples, and documentation to reflect the functional API.

### Migration

The previous `NZTomography` and `TomographyBins` interfaces are no longer supported.

Use the following entry points instead:

- `build_tomography()` for general tomography construction.
- `build_survey_tomography()` for survey-based tomography.
- `build_predefined_tomography()` for externally supplied bins.
- `Tomography` for accessing the resulting distributions and metadata.

This release is not backward-compatible with previous `0.x` releases.

## 0.6.1

### Fixed

- `binny.cosmology.ccl_wrappers` checks at import time if CCL is installed.
- Functions dependent on CCL will tell the user to install CCL if it is not available.
