"""Tomographic redshift distributions and construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from binny.binning.assignments import deterministic_assignment
from binny.binning.core import build_bins
from binny.binning.photoz import photoz_assignment
from binny.binning.specz import specz_assignment
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["Tomography", "build_tomography"]

Assignment = Callable[[FloatArray, FloatArray], FloatArray]
Method = Literal["deterministic", "specz", "photoz"]


@dataclass
class Tomography:
    """Tomographic distributions evaluated on a common redshift grid.

    Args:
        z: True-redshift grid.
        nz: Parent redshift distribution.
        bins: Mapping from bin index to redshift distribution.
        metadata: Numerical metadata and diagnostics associated with the
            tomographic binning. Defaults to an empty dictionary.
        spec: Optional resolved tomography specification, including any
            configuration overrides. None when no specification was used.
        survey_metadata: Optional survey-level metadata, including survey
            properties and provenance. Defaults to an empty dictionary.
    """

    z: FloatArray
    nz: FloatArray
    bins: dict[int, FloatArray]
    metadata: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] | None = None
    survey_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        """Number of tomographic bins."""
        return len(self.bins)

    @property
    def bin_indices(self) -> list[int]:
        """Sorted tomographic bin indices."""
        return sorted(self.bins)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the distribution for a tomographic bin."""
        return self.bins[index]


def build_tomography(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Assignment | None = None,
    *,
    method: Method | None = None,
    params: Mapping[str, Any] | None = None,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize_input: bool = False,
    normalize: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> Tomography:
    """Build tomographic redshift distributions from a bin-assignment model.

    Args:
        z: True-redshift grid.
        nz: Parent redshift distribution.
        edges: Tomographic bin edges.
        assignment: Optional custom assignment function returning an array
            of shape ``(n_bins, n_z)``.
        method: Built-in assignment model: ``"deterministic"``, ``"specz"``,
            or ``"photoz"``. Defaults to ``"deterministic"`` when no
            custom assignment is provided.
        params: Model-specific keyword arguments passed to the selected
            assignment function.
        selection_kind: Whether selections represent probabilities or
            arbitrary non-negative weights.
        normalize_input: Whether to normalize the parent distribution
            before constructing bins.
        normalize: Whether to normalize individual bin distributions.
        norm_method: Numerical integration method for normalization.

    Returns:
        Tomography result containing bin distributions and metadata.

    Raises:
        ValueError: If method and assignment conflict or the method is unknown.
        TypeError: If model parameters are invalid.
    """
    if assignment is not None and method is not None:
        raise ValueError("Provide either assignment or method, not both.")

    if assignment is not None and params is not None:
        raise ValueError("params cannot be used with a custom assignment.")

    selected_method = method or ("custom" if assignment is not None else "deterministic")
    model_params = dict(params) if params is not None else {}

    if assignment is None:
        if selected_method == "deterministic":
            if model_params:
                raise TypeError("Deterministic assignment does not accept model parameters.")
            assignment = deterministic_assignment

        elif selected_method == "specz":

            def assignment(z_axis: FloatArray, bin_edges: FloatArray) -> FloatArray:
                """Evaluate the built-in spectroscopic-redshift assignment."""
                return specz_assignment(z_axis, bin_edges, **model_params)

        elif selected_method == "photoz":

            def assignment(z_axis: FloatArray, bin_edges: FloatArray) -> FloatArray:
                """Evaluate the built-in photometric-redshift assignment."""
                return photoz_assignment(z_axis, bin_edges, **model_params)

        else:
            raise ValueError("method must be 'deterministic', 'specz', or 'photoz'.")

    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    if normalize_input:
        nz_arr = normalize_1d(z_arr, nz_arr, method=norm_method)

    bins, integrals, parent_integral = build_bins(
        z_arr,
        nz_arr,
        edges,
        assignment,
        selection_kind=selection_kind,
        normalize=normalize,
        norm_method=norm_method,
        return_integrals=True,
    )

    fractions = {
        i: area / parent_integral if parent_integral > 0.0 else 0.0 for i, area in integrals.items()
    }

    metadata = {
        "bin_edges": np.asarray(edges, dtype=float).copy(),
        "method": selected_method,
        "params": model_params.copy(),
        "selection_kind": selection_kind,
        "normalize_input": normalize_input,
        "normalize_bins": normalize,
        "norm_method": norm_method,
        "bin_integrals": integrals,
        "parent_integral": parent_integral,
        "frac_per_bin": fractions,
    }

    return Tomography(
        z=z_arr,
        nz=nz_arr,
        bins=bins,
        metadata=metadata,
    )
