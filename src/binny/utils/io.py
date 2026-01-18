"""I/O helpers for binny."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from binny.utils.validators import validate_mixed_segments

__all__ = [
    "load_nz",
    "load_binning_recipe",
    "load_yaml",
]


def load_nz(
    filename: str | Path,
    *,
    x_col: int = 0,
    nz_col: int = 1,
    key: str | None = None,
    delimiter: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a 1D redshift grid z and corresponding n(z) from file.

    Supported formats:
      - .npy  : either shape (N, 2) array (z, n(z)) or separate dict-like with fields
      - .npz  : if `key` is provided, expects an array with shape (N, 2) under that key;
                otherwise tries the first array it finds with shape (N, 2)
      - .txt/.dat/.csv : plain text with at least two columns; `x_col` and `nz_col`
                         control which columns to use.

    The function does NOT normalise n(z); normalisation is handled later by
    ``build_photoz_bins`` / ``build_specz_bins`` via the ``normalize_input`` flag.

    Args:
        filename:
            Path to the file on disk.
        x_col:
            Column index for the redshift axis in text-based files (default: 0).
        nz_col:
            Column index for n(z) in text-based files (default: 1).
        key:
            For .npz files, optional key selecting the stored array. If None,
            the first array with shape (N, 2) is used.
        delimiter:
            Optional delimiter passed to ``np.loadtxt`` for text/csv files.
            If None, numpy will try to guess.

    Returns:
        (z, nz):
            Two 1D arrays with the same length, sorted in ascending z.

    Raises:
        ValueError:
            If the file format is unsupported or content cannot be interpreted
            as a (z, n(z)) pair.
    """
    path = Path(filename)
    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)

        # Case 1: direct (N, 2) array
        if arr.ndim == 2 and arr.shape[1] >= 2:
            z = arr[:, x_col]
            nz = arr[:, nz_col]
        # Case 2: dict-like object saved via np.save (rare, but possible)
        elif arr.dtype.names is not None:
            try:
                z = np.asarray(arr["z"], dtype=float)
                nz = np.asarray(arr["nz"], dtype=float)
            except KeyError as exc:
                raise ValueError(f".npy file {path} does not contain 'z' and 'nz' fields.") from exc
        else:
            raise ValueError(
                f"Unsupported .npy structure in {path!s}; expected an (N, 2) array "
                "or a dict-like with 'z'/'nz'."
            )

    elif ext == ".npz":
        data = np.load(path)

        if key is not None:
            if key not in data:
                raise ValueError(f"Key {key!r} not found in {path!s}.")
            arr = np.asarray(data[key])
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"Array under key {key!r} in {path!s} must have shape (N, 2) "
                    "or have more columns."
                )
            z = arr[:, x_col]
            nz = arr[:, nz_col]
        else:
            # Try to find the first suitable array
            chosen: Any | None = None
            for k in data.files:
                arr = np.asarray(data[k])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    chosen = arr
                    break
            if chosen is None:
                raise ValueError(
                    f"No suitable (N,2) array found in {path!s}. "
                    "Provide a 'key' argument or store a 2-column array."
                )
            z = chosen[:, x_col]
            nz = chosen[:, nz_col]

    elif ext in {".txt", ".dat", ".csv"}:
        arr = np.loadtxt(path, delimiter=delimiter)
        arr = np.asarray(arr)

        if arr.ndim == 1:
            raise ValueError(f"Text file {path!s} must have at least two columns for z and n(z).")
        if arr.shape[1] <= max(x_col, nz_col):
            raise ValueError(
                f"Requested columns x_col={x_col}, nz_col={nz_col} are out of bounds "
                f"for file {path!s} with {arr.shape[1]} columns."
            )

        z = arr[:, x_col]
        nz = arr[:, nz_col]

    else:
        raise ValueError(
            f"Unsupported file extension {ext!r} for {path!s}. "
            "Supported: .npy, .npz, .txt, .dat, .csv."
        )

    # Ensure 1D, finite, and sorted in ascending z
    z = np.asarray(z, dtype=float).ravel()
    nz = np.asarray(nz, dtype=float).ravel()

    if z.size != nz.size:
        raise ValueError(f"z and nz must have the same length; got {z.size} and {nz.size}.")

    if not np.all(np.isfinite(z)):
        raise ValueError("Loaded z contains non-finite values.")
    if not np.all(np.isfinite(nz)):
        raise ValueError("Loaded nz contains non-finite values.")

    # Sort by z in case the file is unordered
    order = np.argsort(z)
    z_sorted = z[order]
    nz_sorted = nz[order]

    return z_sorted, nz_sorted


def load_binning_recipe(path: str) -> list[dict[str, Any]]:
    """Load a mixed-binning recipe from a YAML file.

    Expected YAML structure::

        name: "y10_mixed_scheme"
        n_bins: 5
        segments:
          - method: "eq"            # alias for equidistant
            n_bins: 3
            params:
              x_min: 0.0
              x_max: 1.0
          - method: "equal_number"
            n_bins: 2

    ``params`` is optional; any keys in params are passed through to mixed_edges.
    Arrays like x, weights, z, chi are NOT stored in YAML; they are provided
    at runtime to ``mixed_edges``.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError("Top-level YAML content must be a mapping.")

    segments = data.get("segments")
    if not isinstance(segments, Sequence) or not segments:
        raise ValueError("YAML must contain a non-empty 'segments' sequence.")

    total_n_bins = data.get("n_bins")

    norm_segments: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            raise ValueError(f"Segment {i} must be a mapping.")

        if "method" not in seg or "n_bins" not in seg:
            raise ValueError(f"Segment {i} must contain at least 'method' and 'n_bins'.")

        method = str(seg["method"])
        n_bins = int(seg["n_bins"])
        params = seg.get("params", {}) or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"Segment {i} 'params' must be a mapping if present.")

        norm_segments.append(
            {
                "method": method,
                "n_bins": n_bins,
                "params": dict(params),
            }
        )

    validate_mixed_segments(norm_segments, total_n_bins=total_n_bins)

    return norm_segments


def load_yaml(
    source: str | Path,
    *,
    package: str | None = None,
) -> dict[str, Any]:
    """Loads YAML from disk or from a packaged resource.

    Args:
        source:
            If ``package is None``: treated as a filesystem path.
            If ``package`` is provided: treated as a filename inside that package.
        package:
            Package name for packaged YAML, e.g. ``"binny.surveys.configs"``.
            If None, load from disk.

    Returns:
        Parsed YAML as a top-level mapping.

    Raises:
        FileNotFoundError:
            If the file does not exist.
        ValueError:
            If the YAML root is not a mapping (dict-like).
    """
    if package is None:
        p = Path(source).expanduser()
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        from importlib import resources

        filename = str(source)
        with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        where = str(source) if package is None else f"{package}:{source}"
        got = type(data).__name__
        raise ValueError(f"Top-level YAML content must be a mapping; got {got} in {where}.")

    return dict(data)
