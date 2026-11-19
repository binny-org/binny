"""Tomographic redshift distributions and construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from binny.binning.assignments import deterministic_assignment
from binny.binning.core import build_bins
from binny.binning.edges import resolve_bin_edges, validate_bin_edges
from binny.binning.photoz import photoz_assignment
from binny.binning.specz import _specz_scatter_enabled, specz_assignment
from binny.utils.metadata import finalize_tomo_metadata, save_metadata_txt
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
    edges: FloatArray | None = None,
    assignment: Assignment | None = None,
    *,
    bins: Mapping[str, Any] | None = None,
    method: Method | None = None,
    params: Mapping[str, Any] | None = None,
    z_ph: FloatArray | None = None,
    nz_ph: FloatArray | None = None,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    include_metadata: bool = True,
    include_photoz_diagnostics: bool = False,
    save_metadata_path: str | None = None,
) -> Tomography:
    """Build tomographic redshift distributions from a bin-assignment model.

    Bin edges may be supplied directly or constructed from a declarative
    binning specification. Exactly one of edges or bins must be provided.

    Args:
        z: True-redshift grid.
        nz: Parent redshift distribution.
        edges: Explicit tomographic bin edges. Mutually exclusive with bins.
        assignment: Optional custom assignment function returning an array
            of shape ``(n_bins, n_z)``.
        bins: Declarative binning specification supporting explicit,
            generated, or mixed binning. Mutually exclusive with edges.
        method: Built-in assignment model: ``"deterministic"``, ``"specz"``,
            or ``"photoz"``. Defaults to ``"deterministic"`` when no
            custom assignment is provided.
        params: Model-specific keyword arguments passed to the selected
            assignment function.
        z_ph: Optional observed photometric-redshift grid used to resolve
            bin edges. Requires nz_ph and method="photoz".
        nz_ph: Optional observed photometric-redshift distribution sampled
            on z_ph.
        selection_kind: Whether selections represent probabilities or
            arbitrary non-negative weights.
        normalize_input: Whether to normalize the parent distribution
            before constructing bins. Defaults to True.
        normalize_bins: Whether to normalize individual bin distributions.
            Defaults to True.
        norm_method: Numerical integration method for normalization.
        include_metadata: Whether to include numerical tomography metadata.
        include_photoz_diagnostics: Whether to include photo-z assignment
            probabilities evaluated on the true-redshift grid.
            Requires method="photoz" and also enables numerical metadata.
        save_metadata_path: Optional path for writing tomography metadata
            in human-readable text form. Metadata can be saved independently
            of whether it is returned.

    Returns:
        Tomography result containing bin distributions and metadata.

    Raises:
        ValueError: If the edge inputs, assignment configuration,
            redshift distributions, binning specification, or diagnostic
            options are invalid.
        TypeError: If model parameters are invalid.
    """
    if (edges is None) == (bins is None):
        raise ValueError("Provide exactly one of edges or bins.")

    if (z_ph is None) != (nz_ph is None):
        raise ValueError("Provide both z_ph and nz_ph, or neither.")

    if z_ph is not None and method != "photoz":
        raise ValueError("z_ph and nz_ph require method='photoz'.")

    if assignment is not None and method is not None:
        raise ValueError("Provide either assignment or method, not both.")

    if assignment is not None and params is not None:
        raise ValueError("params cannot be used with a custom assignment.")

    selected_method = method or ("custom" if assignment is not None else "deterministic")
    model_params = dict(params) if params is not None else {}

    if include_photoz_diagnostics and selected_method != "photoz":
        raise ValueError("Photo-z diagnostics require method='photoz'.")

    return_metadata = include_metadata or include_photoz_diagnostics
    need_metadata = return_metadata or save_metadata_path is not None

    assignment_probabilities: dict[int, FloatArray] = {}

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
                selection = photoz_assignment(z_axis, bin_edges, **model_params)

                if include_photoz_diagnostics:
                    assignment_probabilities.update(
                        {i: row.copy() for i, row in enumerate(selection)}
                    )

                return selection

        else:
            raise ValueError("method must be 'deterministic', 'specz', or 'photoz'.")

    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    if normalize_input:
        nz_arr = normalize_1d(z_arr, nz_arr, method=norm_method)

    if z_ph is not None:
        edge_axis, edge_density = validate_axis_and_weights(z_ph, nz_ph)
    else:
        edge_axis, edge_density = z_arr, nz_arr

    if bins is not None:
        edges = resolve_bin_edges(
            axis=edge_axis,
            density=edge_density,
            bins=bins,
            equal_number_axis=z_ph,
            equal_number_weights=nz_ph,
            selection_axis=z_ph,
            selection_density=nz_ph,
            norm_method=norm_method,
            normalize_equal_number_weights=selected_method == "photoz",
        )

    edges = validate_bin_edges(
        edges,
        require_within=(
            (float(z_arr[0]), float(z_arr[-1])) if selected_method == "specz" else None
        ),
    )

    tomo_bins, integrals, parent_integral = build_bins(
        z_arr,
        nz_arr,
        edges,
        assignment,
        selection_kind=selection_kind,
        normalize_bins=normalize_bins,
        norm_method=norm_method,
        return_integrals=True,
    )

    if not need_metadata:
        return Tomography(
            z=z_arr,
            nz=nz_arr,
            bins=tomo_bins,
        )

    inputs: dict[str, Any] = {
        "method": selected_method,
        "params": model_params.copy(),
        "bins": dict(bins) if bins is not None else None,
        "selection_kind": selection_kind,
        "normalize_input": normalize_input,
        "normalize_bins": normalize_bins,
        "norm_method": norm_method,
        "z_ph_provided": z_ph is not None,
        "nz_ph_provided": nz_ph is not None,
    }

    if selected_method == "photoz":
        inputs["bin_edges_space"] = "photoz"

    elif selected_method == "specz":
        scatter_enabled = _specz_scatter_enabled(
            edges.size - 1,
            specz_scatter=model_params.get("specz_scatter"),
            sigma0=model_params.get("sigma0", 0.0),
            sigma1=model_params.get("sigma1", 0.0),
        )

        inputs.update(
            {
                "bin_edges_space": "truez",
                "response_matrix_provided": (model_params.get("response_matrix") is not None),
                "response": (
                    "M_total = M_scatter @ M_cat" if scatter_enabled else "M_total = M_cat"
                ),
            }
        )

    metadata = finalize_tomo_metadata(
        kind=selected_method,
        z=z_arr,
        parent_nz=nz_arr,
        bin_edges=edges,
        bins=tomo_bins,
        inputs=inputs,
        parent_norm=parent_integral,
        bins_norms=integrals,
        include_metadata=True,
        save_metadata_path=None,
    )

    if metadata is not None:
        if include_photoz_diagnostics:
            metadata["photoz_diagnostics"] = {
                "observed_bin_edges": edges.copy(),
                "true_z": z_arr.copy(),
                "assignment_probabilities": assignment_probabilities,
                "description": (
                    "P(i | z) for each observed photo-z bin, evaluated on the true-z grid."
                ),
            }

        if save_metadata_path is not None:
            save_metadata_txt(metadata, save_metadata_path)

    return Tomography(
        z=z_arr,
        nz=nz_arr,
        bins=tomo_bins,
        metadata=metadata if return_metadata and metadata is not None else {},
    )
