"""Survey-level convenience APIs.

This module provides high-level helpers for building tomographic redshift bins
from simple, user-facing inputs.

The goal is to make it easy to go from either:

- a named parent distribution model ``n(z)`` plus a binning recipe, or
- a minimal YAML configuration file,

to a dictionary of tomographic-bin distributions ``{bin_index: n_i(z)}`` sampled
on a common true-redshift grid ``z``.

The functions in this module are *convenience wrappers* built on top of the
lower-level APIs in :mod:`binny.api.distributions` and :mod:`binny.api.tomo`.
They do not enforce a strict schema beyond basic type/shape expectations, and
are intended as an approachable entry point for new users.

Typical usage patterns:

1) Survey-agnostic photo-z or spec-z tomography from a named n(z) model::

    >>> import numpy as np
    >>> from binny.api.surveys import photoz_tomography
    >>> z = np.linspace(0.0, 3.0, 301)
    >>> bins = photoz_tomography(
    ...     z=z,
    ...     nz_model="smail",
    ...     nz_params={"z0": 0.5, "alpha": 2.0, "beta": 1.0},
    ...     binning_scheme="equidistant",
    ...     n_bins=4,
    ...     params={"scatter_scale": 0.05, "mean_offset": 0.01},
    ... )
    >>> sorted(bins)
    [0, 1, 2, 3]

2) Run a minimal, shipped example configuration (no survey-specific presets)::

    >>> from binny.api.surveys import tomo_from_config
    >>> bins_pz = tomo_from_config(config_file="example_minimal_photoz.yaml")
    >>> isinstance(bins_pz, dict)
    True
    >>> bins_sz = tomo_from_config(config_file="example_minimal_specz.yaml")
    >>> isinstance(bins_sz, dict)
    True

3) Survey-specific convenience wrapper (LSST)::

    >>> from binny.api.surveys import lsst_tomography
    >>> z = np.linspace(0.0, 3.0, 101)
    >>> bins = lsst_tomography(z=z, year=1, sample="source")
    >>> isinstance(bins, dict)
    True
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np

from binny.api.config import load_config
from binny.api.distributions import redshift_distribution
from binny.api.tomo import photoz_bins as _photoz_bins
from binny.api.tomo import specz_bins as _specz_bins
from binny.api.tomo import tomo_bins as _tomo_bins
from binny.axes.grids import linear_grid
from binny.surveys.lsst import lsst_tomography as _lsst_tomography

Sample = Literal["lens", "source"]

__all__ = [
    "photoz_tomography",
    "specz_tomography",
    "lsst_tomography",
    "list_configs",
    "config_path",
    "tomo_from_config",
]

_CONFIG_PKG = "binny.surveys.configs"


def list_configs() -> list[str]:
    """Lists shipped YAML configuration files.

    Returns the filenames of YAML configs distributed with Binny under the
    package directory ``binny.surveys.configs``.

    These configs are intended as small, editable examples that users can copy,
    modify, and run through :func:`tomo_from_config`.

    Returns:
        Sorted list of config filenames (e.g. ``[
        "example_minimal_photoz.yaml", ...]``).
    """
    root = resources.files(_CONFIG_PKG)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_file() and p.name.endswith((".yaml", ".yml"))
    )


def config_path(filename: str) -> Path:
    """Resolves a shipped configuration filename to a local filesystem path.

    This helper looks for ``filename`` inside the installed package data under
    ``binny.surveys.configs`` and returns a usable filesystem :class:`~pathlib.Path`.

    Use this when you want to open a shipped config directly, or to implement
    "filename-or-path" behavior in user-facing APIs.

    Args:
        filename: Name of a shipped YAML file (e.g.
        ``"example_minimal_photoz.yaml"``).

    Returns:
        A filesystem path to the shipped YAML file.

    Raises:
        FileNotFoundError: If ``filename`` is not found among shipped configs.
    """
    root = resources.files(_CONFIG_PKG)
    p = root / filename
    if not p.is_file():
        raise FileNotFoundError(
            f"No shipped config named {filename!r}. Available: {list_configs()}"
        )
    with resources.as_file(p) as real_path:
        return Path(real_path)


def _build_z_from_grid_spec(zspec: Mapping[str, Any]) -> np.ndarray:
    """Builds a true-redshift sampling grid from a minimal YAML grid
    specification.

    The returned grid is used as the common true-z axis on which the parent
    distribution ``n(z)`` and all tomographic bins are evaluated.

    The expected mapping is::

        {"start": <float>, "stop": <float>, "n": <int>}

    Args:
        zspec: Mapping containing the keys ``start``, ``stop``, and ``n``.

    Returns:
        One-dimensional ``float64`` NumPy array of length ``n`` spanning
        ``[start, stop]`` (including endpoints).

    Raises:
        ValueError: If required keys are missing or cannot be converted.
    """
    try:
        z_min = float(zspec["start"])
        z_max = float(zspec["stop"])
        n = int(zspec["n"])
    except KeyError as e:
        raise ValueError("grid.z must contain keys: start, stop, n") from e
    return linear_grid(z_min, z_max, n)


def _build_nz_from_config(nz_cfg: Mapping[str, Any], z: np.ndarray) -> np.ndarray:
    """Builds a parent redshift distribution from a minimal YAML specification.

    The YAML config specifies a named distribution model and a parameter mapping.
    This function evaluates that model on the provided grid ``z``.

    Expected structure::

        nz:
          model: <name>
          params: { ... }

    Args:
        nz_cfg: Mapping describing the distribution model and its parameters.
        z: True-redshift grid on which to evaluate the distribution.

    Returns:
        One-dimensional array ``n(z)`` evaluated on ``z`` (``float64``).

    Raises:
        ValueError: If the configuration is missing required fields or if
            ``params`` is not a mapping.
    """
    try:
        model = str(nz_cfg["model"])
    except KeyError as e:
        raise ValueError("nz must contain a 'model' field.") from e

    params = nz_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("nz.params must be a mapping.")
    return redshift_distribution(model, z, **dict(params))


def photoz_tomography(
    *,
    z: Any,
    nz_model: str,
    nz_params: Mapping[str, Any],
    binning_scheme: Any,
    n_bins: int,
    params: Mapping[str, Any] | None = None,
) -> dict[int, np.ndarray]:
    """Build survey-agnostic photo-z tomographic bins from a named parent n(z).

    This function is a convenience wrapper that combines two steps:

    1) Evaluate aDF-l; sorry.

    1) Evaluate a named parent distribution model ``n(z)`` on the provided
       true-redshift grid ``z``.
    2) Construct photo-z tomographic bins using a binning scheme and photo-z
       model parameters (e.g., scatter/outliers), returning per-bin true-z
       distributions.

    Use this when you want a simple, survey-independent way to generate photo-z
    tomography from a compact set of inputs.

    Args:
        z: True-redshift grid on which the distributions are defined.
        nz_model: Name of the parent distribution model (see
            :func:`binny.api.distributions.available_redshift_distributions`).
        nz_params: Parameters forwarded to the parent distribution model.
        binning_scheme: Recipe used to construct observed-redshift bin edges,
            e.g. ``"equidistant"``, ``"equal_number"``, or a mixed/segmented
            scheme supported by the underlying tomo builder.
        n_bins: Number of tomographic bins to construct.
        params: Optional photo-z model and normalization parameters forwarded to
            the underlying photo-z binning implementation.

    Returns:
        Dictionary mapping bin index (``int``) to ``n_i(z)`` arrays sampled on
        the input grid ``z``.

    Examples:
        >>> import numpy as np
        >>> from binny.api.surveys import photoz_tomography
        >>> z = np.linspace(0.0, 3.0, 301)
        >>> bins = photoz_tomography(
        ...     z=z,
        ...     nz_model="smail",
        ...     nz_params={"z0": 0.5, "alpha": 2.0, "beta": 1.0},
        ...     binning_scheme="equidistant",
        ...     n_bins=4,
        ...     params={"scatter_scale": 0.05, "mean_offset": 0.01},
        ... )
        >>> sorted(bins)
        [0, 1, 2, 3]
    """
    z_arr = np.asarray(z, dtype=np.float64)
    nz_arr = redshift_distribution(str(nz_model), z_arr, **dict(nz_params))
    return _photoz_bins(
        z=z_arr,
        nz=nz_arr,
        binning_scheme=binning_scheme,
        n_bins=int(n_bins),
        **dict(params or {}),
    )


def specz_tomography(
    *,
    z: Any,
    nz_model: str,
    nz_params: Mapping[str, Any],
    binning_scheme: Any,
    n_bins: int,
    params: Mapping[str, Any] | None = None,
) -> dict[int, np.ndarray]:
    """Build survey-agnostic spec-z tomographic bins from a named parent n(z).

    This function is a convenience wrapper that combines two steps:

    1) Evaluate a named parent distribution model ``n(z)`` on the provided
       true-redshift grid ``z``.
    2) Construct spec-z tomographic bins using a binning scheme and spec-z
       selection/response parameters, returning per-bin true-z distributions.

    Use this when you want a simple, survey-independent way to generate spec-z
    tomography from a compact set of inputs.

    Args:
        z: True-redshift grid on which the distributions are defined.
        nz_model: Name of the parent distribution model (see
            :func:`binny.api.distributions.available_redshift_distributions`).
        nz_params: Parameters forwarded to the parent distribution model.
        binning_scheme: Recipe used to construct true-redshift bin edges,
            e.g. ``"equidistant"``, ``"equal_number"``, or a mixed/segmented
            scheme supported by the underlying tomo builder.
        n_bins: Number of tomographic bins to construct.
        params: Optional spec-z selection and normalization parameters forwarded
            to the underlying spec-z binning implementation.

    Returns:
        Dictionary mapping bin index (``int``) to ``n_i(z)`` arrays sampled on
        the input grid ``z``.

    Examples:
        >>> import numpy as np
        >>> from binny.api.surveys import specz_tomography
        >>> z = np.linspace(0.0, 2.0, 301)
        >>> bins = specz_tomography(
        ...     z=z,
        ...     nz_model="smail",
        ...     nz_params={"z0": 0.4, "alpha": 2.0, "beta": 1.0},
        ...     binning_scheme="equidistant",
        ...     n_bins=4,
        ... )
        >>> sorted(bins)
        [0, 1, 2, 3]
    """
    z_arr = np.asarray(z, dtype=np.float64)
    nz_arr = redshift_distribution(str(nz_model), z_arr, **dict(nz_params))
    return _specz_bins(
        z=z_arr,
        nz=nz_arr,
        binning_scheme=binning_scheme,
        n_bins=int(n_bins),
        **dict(params or {}),
    )


def tomo_from_config(
    *,
    config_file: str | Path,
    key: str | None = None,
    z: Any | None = None,
) -> dict[int, np.ndarray]:
    """Build tomographic bins from a minimal YAML configuration.

    This function enables a compact, user-editable config-driven workflow:

    - load a YAML config (either a filesystem path or a shipped example filename),
    - build a true-z sampling grid (unless one is provided directly),
    - evaluate a named parent distribution model ``n(z)``,
    - build tomographic bins according to the specified binning kind and recipe.

    The YAML is survey-agnostic and designed for small end-to-end examples.
    It is also a convenient format for tutorials and reproducible notebooks.

    Config resolution:
        If ``config_file`` exists on disk, it is used as a path. Otherwise it is
        interpreted as the filename of a shipped config in ``binny.surveys.configs``.

    Expected minimal schema::

        <top_key>:
          grid:
            z: {start: <float>, stop: <float>, n: <int>}  # used if z is None
          nz:
            model: <distribution name>
            params: { ... }
          tomo:
            kind: photoz|specz
            binning_scheme: <scheme or segments>
            n_bins: <int>
            params: { ... }

    Args:
        config_file: Path to a YAML file, or the filename of a shipped YAML config.
        key: Optional top-level key selecting which config entry to run. If not
            provided, the YAML must contain exactly one top-level entry.
        z: Optional true-redshift grid overriding the grid specification in YAML.

    Returns:
        Dictionary mapping tomographic bin index (``int``) to ``n_i(z)`` arrays
        sampled on the final true-z grid.

    Examples:
        >>> from binny.api.surveys import tomo_from_config
        >>> bins = tomo_from_config(config_file="example_minimal_photoz.yaml")
        >>> isinstance(bins, dict)
        True
    """
    p = Path(config_file)
    if not p.exists():
        p = config_path(str(config_file))

    raw = load_config(p).raw

    if key is None:
        if len(raw) != 1:
            raise ValueError(
                "Config must contain exactly one top-level key, or pass key=..."
            )
        (key,) = raw.keys()

    cfg = raw[key]
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config entry {key!r} must be a mapping.")

    if z is None:
        grid = cfg.get("grid")
        if not isinstance(grid, Mapping):
            raise ValueError(
                "Config must contain a 'grid' mapping when z is not passed."
            )
        zspec = grid.get("z")
        if not isinstance(zspec, Mapping):
            raise ValueError("Config must contain grid.z mapping when z is not passed.")
        z_arr = _build_z_from_grid_spec(zspec)
    else:
        z_arr = np.asarray(z, dtype=np.float64)

    nz_cfg = cfg.get("nz")
    if not isinstance(nz_cfg, Mapping):
        raise ValueError("Config must contain an 'nz' mapping.")
    nz_arr = _build_nz_from_config(nz_cfg, z_arr)

    tomo_cfg = cfg.get("tomo")
    if not isinstance(tomo_cfg, Mapping):
        raise ValueError("Config must contain a 'tomo' mapping.")

    kind = str(tomo_cfg.get("kind", "")).lower()
    binning_scheme = tomo_cfg.get("binning_scheme")
    n_bins = tomo_cfg.get("n_bins")
    if n_bins is None:
        raise ValueError("Config tomo.n_bins is required.")

    params = tomo_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("Config tomo.params must be a mapping if provided.")

    return _tomo_bins(
        kind=kind,
        z=z_arr,
        nz=nz_arr,
        binning_scheme=binning_scheme,
        n_bins=int(n_bins),
        params=dict(params),
    )


def lsst_tomography(
    *,
    z: Any | None = None,
    year: int,
    sample: Sample,
    config_file: str = "lsst_survey_specs.yaml",
    normalize_input: bool = True,
    normalize_bins: bool = True,
) -> dict[int, np.ndarray]:
    """Build LSST tomographic redshift bins for a given sample and survey year.

    This is a survey-specific convenience wrapper that constructs LSST lens or
    source tomographic bins for a chosen survey-year scenario (e.g., year 1 or
    year 10). It is intended for quick access to LSST-like tomography without
    manually wiring together distributions, binning schemes, and photo-z models.

    Args:
        z: True-redshift grid on which the bin distributions are evaluated.
        year: Survey-year/scenario selector (typically 1 or 10).
        sample: Which LSST sample to build. Use ``"source"`` for weak-lensing
            sources or ``"lens"`` for number-count lenses.
        config_file: Name or path of the LSST YAML spec used by the underlying
            implementation.
        normalize_input: If ``True``, normalize the parent distribution before
            binning (behavior defined by the LSST implementation).
        normalize_bins: If ``True``, normalize each tomographic bin distribution
            after binning (behavior defined by the LSST implementation).

    Returns:
        Dictionary mapping tomographic bin index (``int``) to ``n_i(z)`` arrays
        sampled on ``z``.
    """
    return _lsst_tomography(
        z=z,
        year=year,
        sample=sample,
        config_file=config_file,
        normalize_input=normalize_input,
        normalize_bins=normalize_bins,
    )
