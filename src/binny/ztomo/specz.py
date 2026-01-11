"""Functions to build spectroscopic-redshift tomographic bins.

This module constructs true-redshift tomographic-bin distributions for
spectroscopic (true-z) binning, with optional survey-style bin-response effects
that map true bins to observed bins.

The main entry point is :func:`build_specz_bins`, which returns a dictionary
mapping observed bin index -> observed-bin distribution ``n_obs_i(z)``, all
evaluated on a common true-z grid ``z``.

Model overview
--------------
1) True-bin selection (top-hat in true z).

For true bin ``j`` with edges ``[z_j, z_{j+1}]``, define a selection window
``S_j(z)`` (left-closed, right-open by default) scaled by per-bin completeness
``c_j``::

    S_j(z) = c_j * 1_{[z_j, z_{j+1})}(z)

and build the true-bin distribution from the parent ``n(z)``::

    n_true_j(z) = n(z) * S_j(z)

2) Bin-to-bin response (optional).

Observed bins are constructed by mixing the true bins with a column-stochastic
response matrix ``M``::

    M[i, j] = P(i_obs | j_true)
    sum_i M[i, j] = 1  for every true bin j

The observed-bin distributions are::

    n_obs_i(z) = sum_j M[i, j] * n_true_j(z)

Supported response effects
--------------------------
- Catastrophic misassignment (bin-level).
  A per-bin catastrophic fraction ``f_j`` is redistributed from the diagonal to
  other observed bins using a leakage prescription (uniform, neighbor, or
  Gaussian in bin-index space). If an explicit ``misassignment_matrix`` is
  provided, it is used directly and overrides the leakage model.

- Finite spec-z measurement scatter (optional).
  Builds a second response matrix by integrating a Gaussian measurement model
  for ``z_hat | z_true`` over observed bin edges to get::

      P(z_hat in bin_i | z_true)

  then averages within each true bin to obtain a column-stochastic matrix
  ``M_scatter``.

If both effects are enabled, the total response is applied as::

    M_total = M_scatter @ M_cat

meaning catastrophic reassignment is applied first (at the bin level), followed
by additional mixing from measurement scatter.

Normalization
-------------
- If ``normalize_input=True`` (default), the parent ``n(z)`` is normalized to
  integrate to 1 over ``z`` before binning.
- If ``normalize_bins=True`` (default), each output ``n_obs_i(z)`` is normalized
  to integrate to 1 over ``z`` (for non-empty bins).

Examples
--------
Explicit bin edges (true-z space)::

    >>> import numpy as np
    >>> from binny.ztomo.specz import build_specz_bins
    >>> z = np.linspace(0.0, 2.0, 501)
    >>> nz = z**2 * np.exp(-z)
    >>> bin_edges = [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> bins = build_specz_bins(z, nz, bin_edges)
    >>> sorted(bins)
    [0, 1, 2, 3]
    >>> bins[0].shape
    (501,)

Binning scheme + n_bins (edges constructed internally)::

    >>> bins = build_specz_bins(z, nz, binning_scheme="equidistant", n_bins=4)
    >>> sorted(bins)
    [0, 1, 2, 3]

Equal-number in true-z using (z, nz) as weights::

    >>> bins = build_specz_bins(
    ...     z,
    ...     nz,
    ...     binning_scheme="equal_number",
    ...     n_bins=3,
    ...     normalize_input=True,
    ...     normalize_bins=False,
    ... )
    >>> sorted(bins)
    [0, 1, 2]

Mixed / segmented scheme (sequence of segments)::

    >>> segments = [
    ...     {"scheme": "equidistant", "n_bins": 2, "z_min": 0.0, "z_max": 1.0},
    ...     {"scheme": "equidistant", "n_bins": 2, "z_min": 1.0, "z_max": 2.0},
    ... ]
    >>> bins = build_specz_bins(z, nz, binning_scheme=segments, n_bins=None)
    >>> sorted(bins)
    [0, 1, 2, 3]

Mixed / segmented scheme (dict with "segments" key)::

    >>> scheme = {
    ...     "segments": [
    ...         {"scheme": "equidistant", "n_bins": 3, "z_min": 0.0, "z_max": 1.5},
    ...         {"scheme": "equidistant", "n_bins": 1, "z_min": 1.5, "z_max": 2.0},
    ...     ]
    ... }
    >>> bins = build_specz_bins(z, nz, binning_scheme=scheme, n_bins=None)
    >>> sorted(bins)
    [0, 1, 2, 3]

Catastrophic misassignment (neighbor leakage)::

    >>> bins = build_specz_bins(
    ...     z,
    ...     nz,
    ...     bin_edges,
    ...     catastrophic_frac=0.2,
    ...     leakage_model="neighbor",
    ...     normalize_bins=False,
    ... )
    >>> sorted(bins)
    [0, 1, 2, 3]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_n_bins,
    validate_response_matrix,
)
from binny.ztomo.ztomo_utils import mixed_edges

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
    "build_specz_misassignment_matrix",
    "apply_misassignment_matrix",
    "specz_gaussian_misassignment_matrix",
]

FloatArray1D: TypeAlias = NDArray[np.float64]
FloatArray2D: TypeAlias = NDArray[np.float64]
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]


def build_specz_bins(
    z: FloatArray1D,
    nz: FloatArray1D,
    bin_edges: FloatArray1D | None = None,
    *,
    binning_scheme: BinningScheme | None = None,
    n_bins: int | None = None,
    completeness: Sequence[float] | float = 1.0,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    misassignment_matrix: FloatArray2D | None = None,
    specz_scatter: Sequence[float] | float | None = None,
    specz_scatter_model: Literal["const", "sigma0_plus_sigma1_1pz"] = "const",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> dict[int, FloatArray1D]:
    """Builds observed spectroscopic tomographic bins from a parent true-n(z).

    This constructs true-bin distributions via top-hat selection in true redshift,
    then (optionally) mixes those true bins into observed bins with a column-
    stochastic response matrix.

    Construction:
      1) True bins (with completeness ``c_j``):
         ``n_true_j(z) = n(z) * c_j * 1_{[z_j, z_{j+1})}(z)``

      2) Optional response mixing:
         ``n_obs_i(z) = sum_j M[i, j] * n_true_j(z)``,
         where ``M[i, j] = P(i_obs | j_true)`` and ``sum_i M[i, j] = 1``.

    The response can include:
      - Catastrophic bin misassignment (``M_cat``), built from ``catastrophic_frac``
        and ``leakage_model`` unless an explicit ``misassignment_matrix`` is given.
      - Optional measurement scatter (``M_scatter``) from a Gaussian ``z_hat | z``
        model integrated over bin edges.

    If both are enabled, the total response is:
      ``M_total = M_scatter @ M_cat``.

    Bin edges:
      - If ``bin_edges`` is provided, it is used directly (recommended for spec-z).
      - Otherwise, edges are constructed from ``binning_scheme``:
        * String scheme + ``n_bins`` (e.g. "equidistant", "equal_number")
        * Mixed segments (sequence or {"segments": [...]}) passed to ``mixed_edges``.
        Segment dicts must use ``z_min``/``z_max`` (not ``x_min``/``x_max``).

    Normalization:
      - If ``normalize_input=True``, normalize the parent ``n(z)`` before binning.
      - If ``normalize_bins=True``, normalize each output bin to integrate to 1
        on ``z`` (for non-empty bins).

    Args:
        z: One-dimensional true-redshift grid.
        nz: Parent distribution evaluated on ``z``.
        bin_edges: Optional explicit true-z bin edges.
        binning_scheme: Optional scheme for constructing edges when ``bin_edges`` is
            ``None``.
        n_bins: Number of bins when ``binning_scheme`` is a string.
        completeness: Per-bin completeness factor(s) in ``[0, 1]``.
        catastrophic_frac: Per-bin catastrophic reassignment fraction(s) in
            ``[0, 1]``.
        leakage_model: Catastrophic leakage model ("uniform", "neighbor", "gaussian").
        leakage_sigma: Width for Gaussian leakage in bin-index space.
        misassignment_matrix: Optional explicit catastrophic response matrix
            ``M_cat`` that overrides ``catastrophic_frac`` and leakage settings.
        specz_scatter: Optional measurement scatter value(s) for a constant-per-bin
            Gaussian scatter model.
        specz_scatter_model: Scatter parameterization used when ``specz_scatter`` is
            ``None`` (supported: "sigma0_plus_sigma1_1pz").
        sigma0: Additive scatter component for "sigma0_plus_sigma1_1pz".
        sigma1: Redshift-dependent scatter component for "sigma0_plus_sigma1_1pz".
        normalize_input: Whether to normalize the parent ``n(z)`` before binning.
        normalize_bins: Whether to normalize each output bin to unit integral.
        norm_method: Integration method for normalization ("trapezoid" or "simpson").

    Returns:
        Mapping from observed bin index to ``n_obs_i(z)`` on the input grid.

    Raises:
        ValueError: If the edge specification is inconsistent or invalid.
        ValueError: If a response matrix is invalid or not column-stochastic.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    if bin_edges is not None:
        if (binning_scheme is not None) or (n_bins is not None):
            raise ValueError(
                "Provide either bin_edges or (binning_scheme, n_bins), not both."
            )
        bin_edges_arr = np.asarray(bin_edges, dtype=float)

    else:
        if binning_scheme is None:
            raise ValueError("bin_edges is None. You must provide binning_scheme.")

        # Case 1: single scheme (string)
        if isinstance(binning_scheme, str):
            if n_bins is None:
                raise ValueError(
                    "bin_edges is None and binning_scheme is a string. "
                    "You must provide n_bins."
                )
            validate_n_bins(n_bins)

            scheme = binning_scheme.lower()
            if scheme in {"equidistant", "eq", "linear"}:
                x_min = float(z_arr[0])
                x_max = float(z_arr[-1])
                bin_edges_arr = equidistant_edges(x_min, x_max, n_bins)

            elif scheme in {"equal_number", "equipopulated", "en"}:
                # use (z, nz) directly (true-z space)
                w = n_arr
                if normalize_input:
                    total = np.trapezoid(w, x=z_arr)
                    if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
                        w = normalize_1d(z_arr, w, method=norm_method)
                bin_edges_arr = equal_number_edges(z_arr, w, n_bins)

            else:
                raise ValueError(
                    "Unsupported binning_scheme. Supported: "
                    "'equidistant' (eq/linear) and 'equal_number' "
                    "(equipopulated/en)."
                )

        # Case 2: mixed segments (sequence/dict)
        else:
            if n_bins is not None:
                raise ValueError(
                    "In mixed binning mode, set n_bins per segment and leave "
                    "n_bins=None."
                )

            if isinstance(binning_scheme, Mapping):
                if "segments" in binning_scheme:
                    segments = binning_scheme["segments"]
                else:
                    raise ValueError(
                        "Mixed binning dict must contain key 'segments' "
                        "(e.g. {'segments': [ ... ]})."
                    )
            else:
                segments = binning_scheme

            if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
                raise ValueError("Mixed binning requires a sequence of segment dicts.")

            # In spec-z, use (z, nz) for equal-number segments (no z_ph concept)
            bin_edges_arr = mixed_edges(
                segments,
                z_axis=z_arr,
                nz_axis=n_arr,
                z_ph=None,
                nz_ph=None,
                normalize_input=normalize_input,
                norm_method=norm_method,
            )

    if bin_edges_arr.ndim != 1:
        raise ValueError("bin_edges must be 1D.")
    if bin_edges_arr.size < 2:
        raise ValueError("bin_edges must have at least two entries.")
    if not np.all(np.isfinite(bin_edges_arr)):
        raise ValueError("bin_edges must contain only finite values.")
    if np.any(np.diff(bin_edges_arr) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")
    if bin_edges_arr[0] < z_arr[0] or bin_edges_arr[-1] > z_arr[-1]:
        raise ValueError(
            f"bin_edges must lie within z-range [{z_arr[0]}, {z_arr[-1]}], "
            f"got [{bin_edges_arr[0]}, {bin_edges_arr[-1]}]."
        )

    n_bins_eff = int(bin_edges_arr.size - 1)
    validate_n_bins(n_bins_eff)

    if normalize_input:
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    completeness_arr = as_per_bin(completeness, n_bins_eff, "completeness")
    true_bins: dict[int, FloatArray1D] = {}

    for j, (z_min, z_max) in enumerate(
        zip(bin_edges_arr[:-1], bin_edges_arr[1:], strict=False)
    ):
        sel = specz_selection_in_bin(
            z_arr,
            float(z_min),
            float(z_max),
            completeness=float(completeness_arr[j]),
        )
        true_bins[j] = n_arr * sel

    matrix_cat = build_specz_misassignment_matrix(
        n_bins_eff,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        misassignment_matrix=misassignment_matrix,
    )

    matrix_total = matrix_cat

    if specz_scatter is not None or (
        specz_scatter is None
        and specz_scatter_model == "sigma0_plus_sigma1_1pz"
        and (sigma0 > 0.0 or sigma1 > 0.0)
    ):
        matrix_scatter = specz_gaussian_misassignment_matrix(
            z_arr=z_arr,
            bin_edges=bin_edges_arr,
            specz_scatter=specz_scatter,
            model=specz_scatter_model,
            sigma0=sigma0,
            sigma1=sigma1,
        )
        matrix_total = matrix_scatter @ matrix_cat

    bins = apply_misassignment_matrix(true_bins, matrix_total)

    if normalize_bins:
        for i in range(n_bins_eff):
            integral = np.trapezoid(bins[i], z_arr)
            if not np.isclose(integral, 0.0, atol=1e-12):
                bins[i] = normalize_1d(z_arr, bins[i], method=norm_method)

    return bins


def specz_selection_in_bin(
    z: FloatArray1D,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> FloatArray1D:
    """Builds the spectroscopic (true-z) bin selection function on a redshift grid.

    This returns the bin “window” used to extract a tomographic true-redshift bin:
    a top-hat indicator on the provided redshift grid ``z``, scaled by an optional
    completeness factor. Conceptually, it represents the probability that an
    object at true redshift ``z`` is included in this spectroscopic bin selection
    before any bin-to-bin response effects (e.g. catastrophic misassignment or
    measurement scatter) are applied.

    The selection is either left-closed / right-open ``[bin_min, bin_max)`` (default)
    or inclusive on the right edge ``[bin_min, bin_max]`` if ``inclusive_right=True``.

    Args:
        z: One-dimensional redshift grid where the selection is evaluated.
        bin_min: Lower true-redshift edge of the bin.
        bin_max: Upper true-redshift edge of the bin.
        completeness: Multiplicative completeness factor in ``[0, 1]`` applied to the
            bin selection (e.g. to model per-bin spectroscopic success rate).
        inclusive_right: If True, include ``bin_max`` in the selection; otherwise the
            right edge is excluded.

    Returns:
        A float64 array with the same shape as ``z`` containing values in
        ``{0, completeness}``, representing the bin selection window.

    Raises:
        ValueError: If ``completeness`` is not in ``[0, 1]``.
    """
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(np.float64)


def build_specz_misassignment_matrix(
    n_bins: int,
    *,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    misassignment_matrix: FloatArray2D | None = None,
) -> FloatArray2D:
    """Builds a catastrophic-misassignment response matrix between tomographic bins.

    The returned matrix encodes *bin-level* reassignment errors via
    ``M[i, j] = P(i_obs | j_true)``, i.e. the probability that an object whose
    true bin is ``j`` is recorded in observed bin ``i``. Each column is a
    probability distribution (column-stochastic), so applying ``M`` mixes true-bin
    redshift distributions into observed-bin distributions:
    ``n_obs_i(z) = sum_j M[i, j] * n_true_j(z)``.

    By default (no catastrophes), this is the identity matrix. When catastrophic
    leakage is enabled, a per-true-bin catastrophic fraction ``f_j`` is moved out
    of the diagonal entry and redistributed across observed bins according to a
    leakage prescription (uniform, neighbor-only, or Gaussian in bin index).

    Args:
        n_bins: Number of tomographic bins. The returned matrix has shape
            ``(n_bins, n_bins)``.
        catastrophic_frac: Fraction ``f_j`` in ``[0, 1]`` of objects in true bin ``j``
            that are catastrophically assigned to a different observed bin. May be a
            scalar (applied to all bins) or a sequence of length ``n_bins``.
        leakage_model: How catastrophically misassigned objects are distributed across
            observed bins:
            - ``"uniform"``: distribute equally among all *other* bins
            - ``"neighbor"``: distribute to adjacent bins (edge bins renormalized)
            - ``"gaussian"``: distribute in bin-index space with width ``leakage_sigma``
        leakage_sigma: Width parameter for ``leakage_model="gaussian"``. May be a
            scalar or a sequence of length ``n_bins``. Must be > 0.
        misassignment_matrix: Optional explicit response matrix. If provided, it must
            have shape ``(n_bins, n_bins)`` and be column-stochastic, and it overrides
            ``catastrophic_frac`` and the leakage settings.

    Returns:
        A float64 array ``M`` with shape ``(n_bins, n_bins)`` satisfying
        ``M[i, j] = P(i_obs | j_true)`` and column sums equal to 1.

    Raises:
        ValueError: If ``n_bins`` is invalid.
        ValueError: If ``catastrophic_frac`` is outside ``[0, 1]`` or cannot be
            broadcast to length ``n_bins``.
        ValueError: If ``leakage_model`` is not one of the supported options.
        ValueError: If ``leakage_sigma`` is invalid for the Gaussian leakage model.
        ValueError: If ``misassignment_matrix`` has invalid shape or normalization.
    """
    validate_n_bins(n_bins)

    if misassignment_matrix is not None:
        matrix = np.asarray(misassignment_matrix, dtype=float)
        validate_response_matrix(matrix, n_bins)
        return matrix.astype(np.float64, copy=False)

    f = as_per_bin(catastrophic_frac, n_bins, "catastrophic_frac").astype(float)
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("catastrophic_frac must be in [0, 1].")

    matrix = np.eye(n_bins, dtype=float)

    if np.allclose(f, 0.0):
        return matrix

    q = np.zeros((n_bins, n_bins), dtype=float)

    leakage_model = leakage_model.lower()
    match leakage_model:
        case "uniform":
            for j in range(n_bins):
                if n_bins == 1:
                    q[0, 0] = 1.0
                else:
                    q[:, j] = 1.0 / (n_bins - 1)
                    q[j, j] = 0.0

        case "neighbor":
            for j in range(n_bins):
                if n_bins == 1:
                    q[0, 0] = 1.0
                elif j == 0:
                    q[1, 0] = 1.0
                elif j == n_bins - 1:
                    q[n_bins - 2, j] = 1.0
                else:
                    q[j - 1, j] = 0.5
                    q[j + 1, j] = 0.5

        case "gaussian":
            sig = as_per_bin(leakage_sigma, n_bins, "leakage_sigma").astype(float)
            if np.any(sig <= 0.0):
                raise ValueError(
                    "leakage_sigma must be > 0 for leakage_model='gaussian'."
                )

            idx = np.arange(n_bins, dtype=float)
            for j in range(n_bins):
                w = np.exp(-0.5 * ((idx - float(j)) / float(sig[j])) ** 2)
                w[j] = 0.0
                s = w.sum()
                if s <= 0.0:
                    if n_bins == 1:
                        q[0, 0] = 1.0
                    elif j == 0:
                        q[1, 0] = 1.0
                    elif j == n_bins - 1:
                        q[n_bins - 2, j] = 1.0
                    else:
                        q[j - 1, j] = 0.5
                        q[j + 1, j] = 0.5
                else:
                    q[:, j] = w / s

        case _:
            raise ValueError(
                "leakage_model must be 'uniform', 'neighbor', or 'gaussian'."
            )

    for j in range(n_bins):
        matrix[:, j] = (1.0 - f[j]) * matrix[:, j] + f[j] * q[:, j]

    validate_response_matrix(matrix, n_bins)
    return matrix.astype(np.float64, copy=False)


def apply_misassignment_matrix(
    bins: Mapping[int, FloatArray1D],
    matrix: FloatArray2D,
) -> dict[int, FloatArray1D]:
    """Applies a bin-to-bin response matrix to tomographic bin distributions.

    This maps a set of *true-bin* redshift distributions to *observed-bin*
    distributions via a column-stochastic response matrix
    ``M[i, j] = P(i_obs | j_true)``.
    Conceptually, each observed bin is a weighted mixture of the true bins:
    ``n_obs_i(z) = sum_j M[i, j] * n_true_j(z)``. The output preserves the common
    redshift grid of the input distributions.

    Args:
        bins: Mapping from true-bin index ``j`` to the corresponding distribution
            ``n_true_j(z)`` evaluated on a common redshift grid. Must contain keys
            ``0..n_bins-1``.
        matrix: Response matrix with shape ``(n_bins, n_bins)`` where
            ``matrix[i, j] = P(i_obs | j_true)``. Must be column-stochastic (each
            column sums to 1).

    Returns:
        Mapping from observed-bin index ``i`` to the mixed distribution
        ``n_obs_i(z)`` on the same grid.

    Raises:
        ValueError: If ``matrix`` has the wrong shape or is not column-stochastic.
        ValueError: If ``bins`` does not contain exactly the keys ``0..n_bins-1``.
        ValueError: If the bin distributions do not all have the same shape.
    """
    n_bins = int(len(bins))
    validate_response_matrix(matrix, n_bins)

    expected = set(range(n_bins))
    keys = set(bins.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            f"bins must contain exactly keys 0..{n_bins - 1}. "
            f"missing={missing}, extra={extra}"
        )

    shapes = {bins[j].shape for j in range(n_bins)}
    if len(shapes) != 1:
        raise ValueError(
            f"All bin arrays must have the same shape; got {sorted(shapes)}."
        )

    arr = np.stack([bins[j] for j in range(n_bins)], axis=0)
    obs = matrix @ arr

    out: dict[int, FloatArray1D] = {}
    for i in range(n_bins):
        out[i] = obs[i].astype(np.float64, copy=False)
    return out


def specz_gaussian_misassignment_matrix(
    *,
    z_arr: FloatArray1D,
    bin_edges: FloatArray1D,
    specz_scatter: Sequence[float] | float | None,
    model: Literal["const", "sigma0_plus_sigma1_1pz"] = "const",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> FloatArray2D:
    """Builds a Gaussian measurement-scatter response matrix for spec-z binning.

    The returned matrix ``M`` has shape ``(n_bins, n_bins)`` and encodes
    ``M[i, j] = P(i_obs | j_true)``, the probability that an object whose true
    redshift lies in true bin ``j`` is observed (after measurement scatter in
    ``z_hat``) to fall into observed bin ``i``. Each column is normalized to sum
    to 1 (column-stochastic), so applying ``M`` to a stack of true-bin
    distributions mixes them into observed-bin distributions while conserving
    total probability.

    Args:
        z_arr: One-dimensional true-redshift grid used to represent bins and to
            evaluate bin-assignment probabilities.
        bin_edges: One-dimensional array of bin edges defining the ``n_bins`` true
            and observed bins in redshift.
        specz_scatter: Gaussian scatter parameter for the measurement model
            ``z_hat | z``. If provided, it is treated as constant within each true
            bin: a scalar applies to all bins, or a sequence of length ``n_bins``
            sets one scatter value per true bin. If ``None``, the scatter is taken
            from ``model`` and ``(sigma0, sigma1)``.
        model: Scatter parameterization used when ``specz_scatter`` is ``None``.
            Currently supported:
            - ``"sigma0_plus_sigma1_1pz"``: ``sigma(z) = sigma0 + sigma1 * (1 + z)``
            The value ``"const"`` is accepted only when ``specz_scatter`` is given.
        sigma0: Additive component of ``sigma(z)`` when
            ``model="sigma0_plus_sigma1_1pz"``.
        sigma1: Redshift-dependent component of ``sigma(z)`` when
            ``model="sigma0_plus_sigma1_1pz"``.

    Returns:
        A float array with shape ``(n_bins, n_bins)`` giving the column-stochastic
        response matrix ``M[i, j] = P(i_obs | j_true)``.

    Raises:
        ValueError: If ``bin_edges`` does not define a valid number of bins.
        ValueError: If ``specz_scatter`` is provided but contains non-positive
            values.
        ValueError: If ``specz_scatter`` is ``None`` and ``model`` is not
            ``"sigma0_plus_sigma1_1pz"``.
        ValueError: If ``sigma0`` or ``sigma1`` is negative.

    Notes:
        This matrix represents *measurement scatter only* and does not include
        catastrophic misassignment. If both effects are modeled, the measurement
        matrix is typically applied after the catastrophic response.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = int(bin_edges_arr.size - 1)

    if specz_scatter is not None:
        sig_bin = as_per_bin(specz_scatter, n_bins, "specz_scatter").astype(float)
        if np.any(sig_bin <= 0.0):
            raise ValueError("specz_scatter must be > 0 when provided.")
    else:
        if model != "sigma0_plus_sigma1_1pz":
            raise ValueError(
                "If specz_scatter is None, model must be 'sigma0_plus_sigma1_1pz'."
            )
        if sigma0 < 0.0 or sigma1 < 0.0:
            raise ValueError("sigma0 and sigma1 must be >= 0.")
        if sigma0 == 0.0 and sigma1 == 0.0:
            return np.eye(n_bins, dtype=np.float64)

    sqrt2 = np.sqrt(2.0)
    matrix = np.zeros((n_bins, n_bins), dtype=float)

    masks: list[np.ndarray] = []
    for j in range(n_bins):
        zmin, zmax = bin_edges_arr[j], bin_edges_arr[j + 1]
        masks.append((z_arr >= zmin) & (z_arr < zmax))

    for j in range(n_bins):
        mask_j = masks[j]
        if not np.any(mask_j):
            matrix[j, j] = 1.0
            continue

        z_in = z_arr[mask_j]

        if specz_scatter is not None:
            sigma = float(sig_bin[j]) * np.ones_like(z_in)
        else:
            sigma = sigma0 + sigma1 * (1.0 + z_in)

        sigma = np.maximum(sigma, 1e-12)

        for i in range(n_bins):
            a = bin_edges_arr[i]
            b = bin_edges_arr[i + 1]
            t1 = (b - z_in) / (sqrt2 * sigma)
            t0 = (a - z_in) / (sqrt2 * sigma)
            p = 0.5 * (erf(t1) - erf(t0))
            matrix[i, j] = float(np.mean(p))

        colsum = matrix[:, j].sum()
        if colsum <= 0.0:
            matrix[j, j] = 1.0
        else:
            matrix[:, j] /= colsum

    validate_response_matrix(matrix, n_bins)
    return matrix.astype(np.float64, copy=False)
