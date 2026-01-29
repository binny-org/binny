from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import binny.surveys.config_utils as cu
from binny.nz.registry import available_models as _available_nz_models
from binny.nz.registry import nz_model as _nz_model
from binny.nz_tomo.bin_stats import population_stats as _population_stats
from binny.nz_tomo.bin_stats import shape_stats as _shape_stats

__all__ = ["NZTomography"]


def _norm_str(x: Any) -> str:
    return str(x).strip().lower()


class NZTomography:
    """User-facing tomography wrapper with minimal surface area."""

    def __init__(self) -> None:
        self._parent: dict[str, Any] | None = None
        self._state: dict[str, Any] | None = None  # holds tomo_spec, bins, tomo_meta

    # -------------------------
    # n(z) registry convenience
    # -------------------------
    @staticmethod
    def nz_model(name: str, z: Any, /, **params: Any) -> np.ndarray:
        return _nz_model(name, z, **params)

    @staticmethod
    def available_nz_models() -> list[str]:
        return _available_nz_models()

    # -------------------------
    # Public API
    # -------------------------
    def clear(self) -> None:
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
        """
        Build bins from config OR mapping OR arrays.

        Returns a payload dict with:
          z, nz, spec, bins, tomo_meta, survey_meta
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

        # 6) Return payload
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
        """One-shot for shipped presets; returns payload dict."""
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

        payload = self.build_bins(
            config_file=config_file,
            role=role,
            year=year,
            z=z,
            overrides=overrides,
            include_survey_metadata=include_survey_metadata,
            include_tomo_metadata=include_tomo_metadata,
        )
        payload["survey"] = _norm_str(survey)

        if include_stats:
            payload["shape_stats"] = self.shape_stats()
            payload["population_stats"] = self.population_stats()

        return payload

    # -------------------------
    # Stats API
    # -------------------------
    def bins(self) -> Mapping[int, np.ndarray]:
        self._require_bins()
        return self._state["bins"]

    def tomo_meta(self) -> Mapping[str, Any] | None:
        self._require_state()
        return self._state.get("tomo_meta")

    def shape_stats(self, **kwargs: Any) -> dict[str, Any]:
        self._require_bins()
        return _shape_stats(z=self._parent["z"], bins=self._state["bins"], **kwargs)

    def population_stats(self, **kwargs: Any) -> dict[str, Any]:
        self._require_bins()
        meta = self._state.get("tomo_meta")
        if meta is None:
            raise ValueError("No tomo metadata cached. Rebuild with include_tomo_metadata=True.")
        return _population_stats(bins=self._state["bins"], metadata=meta, **kwargs)

    # -------------------------
    # Internal plumbing
    # -------------------------
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
        # arrays path
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

        # config path -> resolve to mapping, then continue
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
        if self._parent is None or self._state is None or self._state.get("tomo_spec") is None:
            raise ValueError("No cached entry. Call build_bins(...) first.")

    def _require_bins(self) -> None:
        self._require_state()
        if self._state.get("bins") is None:
            raise ValueError("No bins cached. Call build_bins(...) first.")

    def _resolve_builder(self, kind: str):
        k = _norm_str(kind)
        if k == "photoz":
            from binny.nz_tomo.photoz import build_photoz_bins

            return build_photoz_bins
        if k == "specz":
            from binny.nz_tomo.specz import build_specz_bins

            return build_specz_bins
        raise ValueError(f"Unknown tomography kind {k!r}.")
