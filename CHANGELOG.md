# Changelog

## 2.0.0 - 2026-11-19

### Breaking changes

- Reorganized the public API into seven primary namespaces: `nz`, `binning`, `tomography`, `calibration`, `analysis`, `surveys`, and `style`.
- Unified deterministic, photometric, spectroscopic, and custom selection-based tomography construction through `build_tomography()`.
- Removed redundant construction functions, obsolete aliases, and previous module paths.
- Standardized normalization arguments across tomography construction interfaces.
- Introduced a consistent nested metadata structure.
- Removed backward compatibility with superseded public interfaces.

### Added

- Dedicated `binny.calibration`, `binny.analysis`, and `binny.style` namespaces.
- Unified probability-based and weight-based selection functionality.
- Consistent metadata and photo-z assignment diagnostics across construction workflows.
- Improved serialization of large and multidimensional metadata arrays.
- Public API validation and comprehensive regression tests.
- Dedicated package architecture documentation.

### Changed

- Consolidated tomography construction from direct arrays, binning specifications, and survey configurations into a shared functional pipeline.
- Separated bin-edge resolution, redshift assignment, distribution construction, and metadata generation.
- Relocated calibration, distribution analysis, and plotting utilities into dedicated namespaces.
- Centralized normalization, integration, validation, and distribution-comparison utilities.
- Standardized parent distribution and tomographic bin normalization conventions.
- Preserved pre-normalization bin integrals for population fraction calculations.
- Simplified plotting palettes and colormap interfaces.
- Moved `cmasher` from runtime dependencies to documentation dependencies.
- Migrated survey presets, benchmarks, examples, and documentation to the reorganized API.
- Extensively expanded and reorganized theoretical documentation.

### Numerical improvements

- Replaced grid-dependent spectroscopic measurement-scatter averaging with continuous integration over true-redshift bin intervals.
- Made spectroscopic Gaussian response matrices independent of redshift grid resolution and spacing.
- Improved numerical integration of narrow Gaussian measurement-scatter distributions.
- Ensured consistent quadrature conventions throughout tomography construction and metadata generation.
- Improved validation of bin edges, selection probabilities, selection weights, and normalization inputs.

### Fixed

- Inconsistent normalization and integration behavior across tomography construction methods.
- Inconsistent parent distribution handling and population fraction calculations.
- Numerical sensitivity of spectroscopic scatter calculations to redshift grid sampling.
- Inconsistent metadata structures and serialization of NumPy arrays.
- Outdated spectroscopic scatter model identifiers in packaged survey configurations.
- Duplicate implementations and redundant public entry points.

### Migration

This release is not backward-compatible with previous `1.x` releases.

Use the following primary namespaces:

- `binny.nz` for parent redshift distributions, distribution models, and mock catalog sampling.
- `binny.binning` for binning schemes, bin edges, and redshift assignment models.
- `binny.tomography` for tomographic distribution construction and externally supplied bins.
- `binny.calibration` for redshift distribution calibration, photo-z error fitting, and PSF-dependent selection.
- `binny.analysis` for distribution statistics, comparisons, sample composition, and bin combination filtering.
- `binny.surveys` for survey configurations and presets.
- `binny.style` for plotting palettes, colormaps, and visualization utilities.

Use `build_tomography()` with the appropriate assignment method instead of the removed specialized photo-z and spec-z builders.

Use `normalize_input` and `normalize_bins` to control parent and individual bin normalization independently where supported.

Existing workflows that access tomography metadata must be updated to use the new nested metadata structure.

### Testing and validation

- Expanded unit, regression, invariance, and public API tests.
- More than 28,000 passing tests, including the full slow-test suite.
- Branch-aware test coverage exceeding 95%.
- Passing documentation tests, documentation builds, and Ruff checks.

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
