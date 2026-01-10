from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

Sample = Literal["lens", "source"]


def _load_npy(path: Path) -> Any:
    """Load .npy that may contain dicts/object arrays."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    return obj


def _stack_bins(obj: Any) -> np.ndarray:
    """Convert bins to (n_bins, n_z) float array."""
    if isinstance(obj, dict):
        return np.stack([np.asarray(obj[k], dtype=float) for k in sorted(obj)], axis=0)

    arr = np.asarray(obj)
    if arr.dtype == object:
        return np.stack([np.asarray(x, dtype=float) for x in arr], axis=0)

    return np.asarray(arr, dtype=float)


def load_benchmark(*, preset: str, sample: Sample, year: int) -> dict[str, Any]:
    """Load one reference set for a given preset/sample/year."""
    # Anchor to repo root via this file location
    repo_root = Path(__file__).resolve().parents[1]  # if this file is in tests/
    d = repo_root / "tests" / "reference" / "data" / preset

    if year not in (1, 10):
        raise ValueError("year must be one of {1, 10} for these reference files.")

    ytag = f"Y{year}"
    z = _load_npy(d / f"redshift_range_{ytag}.npy")
    bins_obj = _load_npy(d / f"{sample}_bins_{ytag}.npy")

    return {
        "z": np.asarray(z, dtype=float),
        "bins": _stack_bins(bins_obj),
    }
