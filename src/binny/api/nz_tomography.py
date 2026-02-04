"""Tools for building tomographic redshift bins.

This module provides a simple interface for constructing tomographic redshift
bins using either photometric or spectroscopic redshifts. Bins can be built
from survey presets, configuration mappings, or directly from user-supplied
redshift distributions.

Once bins are built, the results are kept internally so that related quantities
such as population statistics or bin-to-bin comparisons can be computed without
repeating the setup.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import binny.nz_tomo.bin_similarity as _bin_sim
import binny.surveys.config_utils as cu
from binny.nz.registry import available_models as _available_nz_models
from binny.nz.registry import nz_model as _nz_model
from binny.nz_tomo.bin_stats import population_stats as _population_stats
from binny.nz_tomo.bin_stats import shape_stats as _shape_stats

__all__ = ["NZTomography"]


class NZTomography:
    """User-facing tomography wrapper with minimal surface area."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._parent: dict[str, Any] | None = None
        self._state: dict[str, Any] | None = None

    @staticmethod
    def nz_model(name: str, z: Any, /, **params: Any) -> np.ndarray:
        """Evaluates a registered parent redshift-distribution model.

        This is a convenience wrapper around the n(z) registry. It evaluates the named
        model on the provided redshift grid and returns the resulting array.

        Args:
            name: Name of a registered n(z) model.
            z: Redshift grid (array-like) where the model is evaluated.
            **params: Model-specific parameters forwarded to the registered implementation.

        Returns:
            Array of model values evaluated on ``z``.

        Raises:
            ValueError: If the model name is unknown or the provided parameters are invalid
                for the requested model.
        """
        return _nz_model(name, z, **params)

    @staticmethod
    def list_nz_models() -> list[str]:
        """List available registered n(z) model names.

        Returns the set of model identifiers currently registered with the n(z) registry.
        This is useful for discovery and for validating configuration inputs.

        Returns:
            A sorted list of registered n(z) model names.
        """
        return _available_nz_models()

    @staticmethod
    def list_surveys() -> list[str]:
        """List available shipped survey preset names.

        Survey presets are YAML configuration files shipped with the package, following
        the naming convention ``<preset>_survey_specs.yaml``. This method returns the
        available preset base names (without the suffix), suitable for passing to
        :meth:`build_survey_bins`.

        Returns:
            A sorted list of available survey preset base names.
        """
        presets: list[str] = []
        for name in cu.list_configs():
            s = str(name)
            if s.endswith("_survey_specs.yaml"):
                presets.append(s.removesuffix("_survey_specs.yaml"))
        return sorted(set(presets))

    def clear(self) -> None:
        """Clears the internal cache."""
        self._parent = None
        self._state = None

    def build_bins(
        self,
        *,
        config_file: str | Path | None = None,
        cfg: Mapping[Any, Any] | None = None,
        z: Any | None = None,
        nz: Any | None = None,
        tomo_spec: Mapping[str, Any] | None = None,
        key: str | None = None,
        role: str | None = None,
        year: Any | None = None,
        kind: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        include_survey_metadata: bool = False,
        include_tomo_metadata: bool = False,
    ) -> dict[str, Any]:
        """Build tomographic bins from a config, mapping, or explicit arrays.

        Given either a survey configuration or direct ``(z, nz)`` arrays, this method
        constructs the corresponding tomographic bins and returns the resulting bin
        distributions along with the relevant inputs.

        Exactly one of the following input modes must be used:
        - ``config_file=...`` (optionally with ``key=...``),
        - ``cfg=...`` (an in-memory config mapping),
        - ``(z=..., nz=..., tomo_spec=...)`` (explicit arrays plus a spec mapping).

        Args:
            config_file: Path to a YAML config file. If provided, the tomography entry is
                selected from the file (optionally using ``key``).
            cfg: In-memory config mapping using the same schema as the YAML files.
            z: True-redshift grid. Used only in the explicit arrays mode.
            nz: Parent distribution evaluated on ``z``. Used only in the explicit arrays mode.
            tomo_spec: Tomography specification mapping for the explicit arrays mode.
            key: Optional entry key when selecting an item from a config file.
            role: Optional selector for a tomography entry (e.g., "lens" or "source").
            year: Optional selector for a tomography entry (e.g., "1", "10").
            kind: Optional tomography kind override ("photoz" or "specz").
            overrides: Optional mapping of values merged into the resolved tomography spec.
                Nested mappings are merged shallowly at the first level.
            include_survey_metadata: Whether to include survey-level metadata in the output
                when building from a config file or mapping.
            include_tomo_metadata: Whether to request tomography metadata from the builder.

        Returns:
            A dictionary with keys:
            - ``"z"``: the true-redshift grid,
            - ``"nz"``: the parent distribution on ``z``,
            - ``"spec"``: the resolved tomography spec used for the build,
            - ``"bins"``: mapping from bin index to bin distribution arrays,
            - ``"tomo_meta"``: tomography metadata if requested, otherwise ``None``,
            - ``"survey_meta"``: survey metadata if requested, otherwise ``None``.

        Raises:
            ValueError: If inputs are inconsistent, no unique entry can be selected, the
                tomography spec is invalid, or the requested kind is unsupported.
        """
        self.clear()

        # 1) Load parent + initial spec into cache
        self._parent, self._state = self._load_parent_and_spec(
            config_file=config_file,
            cfg=cfg,
            z=z,
            nz=nz,
            tomo_spec=tomo_spec,
            key=key,
            role=role,
            year=year,
            include_survey_metadata=include_survey_metadata,
        )

        # 2) Prepare spec (copy + overrides)
        spec = dict(self._state["tomo_spec"])

        if kind is not None:
            spec["kind"] = _norm_str(kind)
        else:
            spec["kind"] = _norm_str(spec.get("kind", "photoz"))

        if overrides:
            for k, v in overrides.items():
                if isinstance(v, Mapping) and isinstance(spec.get(k), Mapping):
                    spec[k] = {**spec[k], **v}
                else:
                    spec[k] = v

        if "bins" not in spec or not isinstance(spec["bins"], Mapping):
            raise ValueError("tomo_spec must contain a 'bins' mapping.")

        # 3) Resolve builder
        builder = self._resolve_builder(spec["kind"])

        # 4) Build bins (schema -> builder kwargs)
        builder_kwargs = cu._builder_kwargs_from_spec(spec)
        out = builder(
            z=self._parent["z"],
            nz=self._parent["nz"],
            include_metadata=include_tomo_metadata,
            **builder_kwargs,
        )

        if include_tomo_metadata:
            bins, tomo_meta = out
        else:
            bins, tomo_meta = out, None

        # 5) Cache state
        self._state["bins"] = bins
        self._state["tomo_meta"] = tomo_meta
        self._state["tomo_spec"] = spec

        # 6) Return output
        return {
            "z": self._parent["z"],
            "nz": self._parent["nz"],
            "spec": dict(spec),
            "bins": bins,
            "tomo_meta": tomo_meta,
            "survey_meta": self._parent.get("survey_meta") if include_survey_metadata else None,
        }

    def build_survey_bins(
        self,
        survey: str,
        *,
        role: str | None = None,
        year: Any | None = None,
        z: Any | None = None,
        overrides: Mapping[str, Any] | None = None,
        include_survey_metadata: bool = False,
        include_tomo_metadata: bool = False,
        include_stats: bool = False,
        config_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build tomographic bins from a shipped survey preset.

        This method loads the configuration for a given survey preset, builds the
        corresponding tomographic bins, and returns the resulting bin distributions.

        Args:
            survey: Survey preset name (case-insensitive), resolved to a shipped YAML file.
            role: Optional selector for a tomography entry (e.g., "lens" or "source").
            year: Optional selector for a tomography entry (e.g., "1", "10").
            z: Optional override for the config-defined redshift grid.
            overrides: Optional mapping merged into the resolved tomography spec.
            include_survey_metadata: Whether to include survey-level metadata in the output.
            include_tomo_metadata: Whether to include tomography metadata in the output.
            include_stats: Whether to compute and include shape/population statistics.
            config_file: Optional explicit path to a survey-spec YAML file. If provided, it
                is used instead of resolving a shipped preset.

        Returns:
            A output dictionary as returned by :meth:`build_bins`, with additional keys:
            - ``"survey"``: normalized survey name,
            - ``"shape_stats"`` and ``"population_stats"`` when ``include_stats=True``.

        Raises:
            FileNotFoundError: If the requested preset name does not resolve to a shipped file.
            ValueError: If the selected entry is invalid or required metadata is unavailable.
        """
        if config_file is None:
            s = _norm_str(survey)
            filename = f"{s}_survey_specs.yaml"
            try:
                config_file = cu.config_path(filename)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Unknown shipped survey preset {survey!r}. "
                    f"Expected {filename!r}. Available: {cu.list_configs()}"
                ) from e

        # If user wants stats, we must have tomo_meta.
        if include_stats:
            include_tomo_metadata = True

        output = self.build_bins(
            config_file=config_file,
            role=role,
            year=year,
            z=z,
            overrides=overrides,
            include_survey_metadata=include_survey_metadata,
            include_tomo_metadata=include_tomo_metadata,
        )
        output["survey"] = _norm_str(survey)

        if include_stats:
            output["shape_stats"] = self.shape_stats()
            output["population_stats"] = self.population_stats()

        return output

    def shape_stats(self, **kwargs: Any) -> dict[str, Any]:
        """Compute shape statistics for the cached bins.

        This is a thin wrapper around :func:`binny.nz_tomo.bin_stats.shape_stats` using
        the cached parent redshift grid and bins from the most recent build.

        Args:
            **kwargs: Additional keyword arguments forwarded to the underlying stats
                function.

        Returns:
            A dictionary of shape statistics computed from the cached bins.

        Raises:
            ValueError: If bins have not been built yet.
        """
        self._require_bins()
        return _shape_stats(z=self._parent["z"], bins=self._state["bins"], **kwargs)

    def population_stats(self, **kwargs: Any) -> dict[str, Any]:
        """Compute population statistics for the cached bins.

        Population statistics require tomography metadata (e.g., per-bin raw integrals
        and parent normalization). This method uses the cached bins and cached metadata
        from the most recent build.

        Args:
            **kwargs: Additional keyword arguments forwarded to the underlying stats
                function.

        Returns:
            A dictionary of population statistics computed from the cached bins and
            cached tomography metadata.

        Raises:
            ValueError: If bins have not been built, or if tomography metadata is not
                available (rebuild with ``include_tomo_metadata=True``).
        """
        self._require_bins()
        meta = self._state.get("tomo_meta")
        if meta is None:
            raise ValueError("No tomo metadata cached. Rebuild with include_tomo_metadata=True.")
        return _population_stats(bins=self._state["bins"], metadata=meta, **kwargs)

    def cross_bin_stats(
        self,
        *,
        overlap: Mapping[str, Any] | None = None,
        pairs: Mapping[str, Any] | None = None,
        leakage: Mapping[str, Any] | None = None,
        pearson: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute a bundle of cross-bin similarity and diagnostic matrices.

        This method operates on the cached bins and returns any requested matrices,
        skipping sections whose argument is ``None``.

        Args:
            overlap: Keyword arguments for overlap metrics, or ``None`` to skip.
            pairs: Keyword arguments for pairwise overlap summaries, or ``None`` to skip.
            leakage: Keyword arguments for leakage matrix computation. Must include
                ``"bin_edges"`` (the bin edges corresponding to the bins), or ``None`` to skip.
            pearson: Keyword arguments for Pearson correlation computation, or ``None`` to skip.

        Returns:
            A dictionary containing a subset of keys ``"overlap"``, ``"correlations"``,
            ``"leakage"``, and ``"pearson"``.

        Raises:
            ValueError: If bins are not cached, or if leakage is requested without
                providing ``bin_edges``.
        """
        self._require_bins()
        z = self._parent["z"]
        bins = self._state["bins"]

        out: dict[str, Any] = {}

        if overlap is not None:
            out["overlap"] = _bin_sim.bin_overlap(z, bins, **dict(overlap))

        if pairs is not None:
            out["correlations"] = _bin_sim.overlap_pairs(z, bins, **dict(pairs))

        if leakage is not None:
            kw = dict(leakage)
            if "bin_edges" not in kw:
                raise ValueError("leakage requires leakage={'bin_edges': ...}.")
            bin_edges = kw.pop("bin_edges")
            out["leakage"] = _bin_sim.leakage_matrix(z, bins, bin_edges, **kw)

        if pearson is not None:
            out["pearson"] = _bin_sim.pearson_matrix(z, bins, **dict(pearson))

        return out

    def _load_parent_and_spec(
        self,
        *,
        config_file: str | Path | None,
        cfg: Mapping[Any, Any] | None,
        z: Any | None,
        nz: Any | None,
        tomo_spec: Mapping[str, Any] | None,
        key: str | None,
        role: str | None,
        year: Any | None,
        include_survey_metadata: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve the parent distribution and tomography specification.

        This internal helper unifies the three supported input modes:
        (1) explicit arrays plus a tomography spec,
        (2) a config file path (optionally selecting a key),
        (3) an in-memory config mapping.

        It returns two dictionaries:
        - a parent bundle containing ``z``, ``nz``, and optional survey metadata,
        - a state bundle containing the parsed tomography spec and empty cache slots.

        Args:
            config_file: Optional YAML config path.
            cfg: Optional in-memory config mapping.
            z: Optional true-redshift grid (arrays mode).
            nz: Optional parent distribution on ``z`` (arrays mode).
            tomo_spec: Optional tomography spec mapping (arrays mode).
            key: Optional config entry key for selection.
            role: Optional selector for tomography entry matching.
            year: Optional selector for tomography entry matching.
            include_survey_metadata: Whether to compute and include survey metadata when
                building from a config mapping.

        Returns:
            A tuple ``(parent, state)`` where:
            - ``parent`` contains keys ``"z"``, ``"nz"``, and ``"survey_meta"``,
            - ``state`` contains keys ``"tomo_spec"``, ``"bins"``, and ``"tomo_meta"``.

        Raises:
            ValueError: If no valid input mode is provided or selection is ambiguous.
        """
        if z is not None and nz is not None and tomo_spec is not None:
            parent = {
                "z": np.asarray(z, dtype=float),
                "nz": np.asarray(nz, dtype=float),
                "survey_meta": None,
            }

            # If user passes arrays, allow tomo_spec without an nz block.
            spec_in = dict(tomo_spec)
            spec_in.setdefault("nz", {"model": "arrays"})  # satisfy _parse_entry

            spec = cu._parse_entry(spec_in)
            spec["kind"] = _norm_str(spec.get("kind", "photoz"))

            state = {"tomo_spec": spec, "bins": None, "tomo_meta": None}
            return parent, state

        if config_file is not None:
            cfg_map, resolved_key = cu._resolve_config_entry(config_file=config_file, key=key)
            return self._load_parent_and_spec(
                config_file=None,
                cfg=cfg_map,
                z=z,
                nz=nz,
                tomo_spec=tomo_spec,
                key=resolved_key,
                role=role,
                year=year,
                include_survey_metadata=include_survey_metadata,
            )

        # mapping path
        if cfg is not None:
            cfg = cu._require_mapping(cfg, what="cfg")
            z_arr = cu._extract_z_grid(cfg, z)

            entries = cu._iter_tomography_entries(cfg)
            matches = cu._select_entries(entries, role=role, year=year)
            entry = cu._require_single(matches, what="tomography entry")

            spec = cu._parse_entry(entry)
            spec["kind"] = _norm_str(spec.get("kind", "photoz"))

            nz_arr = cu._build_parent_nz(entry, z_arr)

            parent = {
                "z": z_arr,
                "nz": nz_arr,
                "survey_meta": (
                    cu._survey_meta(
                        cfg=cfg,
                        resolved_key=str(key or "survey"),
                        role=spec["role"],
                        year=spec["year"],
                    )
                    if include_survey_metadata
                    else None
                ),
            }
            state = {"tomo_spec": spec, "bins": None, "tomo_meta": None}
            return parent, state

        raise ValueError(
            "Provide either (config_file=...), (cfg=...), or (z=..., nz=..., tomo_spec=...)."
        )

    def _require_state(self) -> None:
        """Ensure that a tomography entry has been resolved and cached.

        Raises:
            ValueError: If no cached entry is available (call :meth:`build_bins` first).
        """
        if self._parent is None or self._state is None or self._state.get("tomo_spec") is None:
            raise ValueError("No cached entry. Call build_bins(...) first.")

    def _require_bins(self) -> None:
        """Ensure that bins have been built and cached.

        Raises:
            ValueError: If no cached entry or bins are available
                (call :meth:`build_bins` first).
        """
        self._require_state()
        if self._state.get("bins") is None:
            raise ValueError("No bins cached. Call build_bins(...) first.")

    def _resolve_builder(self, kind: str):
        """Resolve the bin builder callable for a tomography kind.

        Args:
            kind: Tomography kind identifier (e.g., "photoz" or "specz"), case-insensitive.

        Returns:
            A callable builder function (e.g., :func:`build_photoz_bins` or
            :func:`build_specz_bins`).

        Raises:
            ValueError: If ``kind`` is not recognized.
        """
        k = _norm_str(kind)
        if k == "photoz":
            from binny.nz_tomo.photoz import build_photoz_bins

            return build_photoz_bins
        if k == "specz":
            from binny.nz_tomo.specz import build_specz_bins

            return build_specz_bins
        raise ValueError(f"Unknown tomography kind {k!r}.")


def _norm_str(x: Any) -> str:
    """Normalize a value for case-insensitive comparisons.

    Converts the input to ``str``, strips leading/trailing whitespace, and lowers
    the result. This is used to make user inputs (e.g., kind, survey names) robust
    to small formatting differences.

    Args:
        x: Value to normalize.

    Returns:
        A normalized lowercase string.
    """
    return str(x).strip().lower()
