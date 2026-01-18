"""Session-style tomography facade.

Users only import NZTomography. The class caches (z, nz) and an optional
tomography entry spec, then builds bins and computes diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from binny.nz_tomo.bin_stats import population_stats as _population_stats
from binny.nz_tomo.bin_stats import shape_stats as _shape_stats
from binny.surveys.session_core import (
    build_bins_from_state as _build_bins_from_state,
)
from binny.surveys.session_core import (
    load_entry_from_config as _load_entry_from_config,
)
from binny.surveys.session_core import (
    load_entry_from_mapping as _load_entry_from_mapping,
)
from binny.surveys.session_core import (
    make_parent_from_arrays as _make_parent_from_arrays,
)

__all__ = ["NZTomography"]


class NZTomography:
    """User-facing API for tomography: cache parent, build bins, compute stats."""

    def __init__(self) -> None:
        self._parent: dict[str, Any] | None = None
        self._last: dict[str, Any] | None = None

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_config(
        cls,
        config_file: str | Path,
        *,
        key: str | None = None,
        role: str | None = None,
        year: Any | None = None,
        z: Any | None = None,
        include_survey_metadata: bool = False,
    ) -> NZTomography:
        """Create a session by loading one entry from a YAML config."""
        t = cls()
        t.load_entry_from_config(
            config_file=config_file,
            key=key,
            role=role,
            year=year,
            z=z,
            include_survey_metadata=include_survey_metadata,
        )
        return t

    @classmethod
    def from_mapping(
        cls,
        cfg: Mapping[Any, Any],
        *,
        key: str = "survey",
        role: str | None = None,
        year: Any | None = None,
        z: Any | None = None,
        include_survey_metadata: bool = False,
    ) -> NZTomography:
        """Create a session by loading one entry from a mapping."""
        t = cls()
        t.load_entry_from_mapping(
            cfg=cfg,
            key=key,
            role=role,
            year=year,
            z=z,
            include_survey_metadata=include_survey_metadata,
        )
        return t

    # -------------------------
    # State
    # -------------------------
    def clear(self) -> None:
        """Clears cached parent, cached entry spec, and cached bins."""
        self._parent = None
        self._last = None

    def has_parent(self) -> bool:
        """Returns True if a parent (z, nz) has been cached."""
        return self._parent is not None

    def has_entry(self) -> bool:
        """Returns True if an entry spec has been cached."""
        return self._last is not None and self._last.get("tomo_spec") is not None

    def has_bins(self) -> bool:
        """Returns True if bins have been built and cached."""
        return self._last is not None and self._last.get("bins") is not None

    def parent_source(self) -> Literal["arrays", "config", "mapping", "none"]:
        """Returns the cached parent source type."""
        if self._parent is None:
            return "none"
        return self._parent["source"]

    # -------------------------
    # Load / set
    # -------------------------
    def set_parent_from_arrays(
        self,
        *,
        z: Any,
        nz: Any,
        survey_meta: Mapping[str, Any] | None = None,
    ) -> None:
        """Set parent (z, nz) directly. Clears cached entry and bins."""
        self._parent = _make_parent_from_arrays(z=z, nz=nz, survey_meta=survey_meta)
        self._last = None

    def load_entry_from_config(
        self,
        *,
        config_file: str | Path,
        key: str | None = None,
        role: str | None = None,
        year: Any | None = None,
        z: Any | None = None,
        include_survey_metadata: bool = False,
    ) -> None:
        """Load (z, nz, tomo_spec) for one entry from YAML and cache it."""
        self._parent, self._last = _load_entry_from_config(
            config_file=config_file,
            key=key,
            role=role,
            year=year,
            z=z,
            include_survey_metadata=include_survey_metadata,
        )

    def load_entry_from_mapping(
        self,
        *,
        cfg: Mapping[Any, Any],
        key: str = "survey",
        role: str | None = None,
        year: Any | None = None,
        z: Any | None = None,
        include_survey_metadata: bool = False,
    ) -> None:
        """Load (z, nz, tomo_spec) for one entry from a mapping and cache it."""
        self._parent, self._last = _load_entry_from_mapping(
            cfg=cfg,
            key=key,
            role=role,
            year=year,
            z=z,
            include_survey_metadata=include_survey_metadata,
        )

    # -------------------------
    # Accessors
    # -------------------------
    def z(self) -> np.ndarray:
        """Returns cached true-z grid."""
        self._require_parent()
        return self._parent["z"]

    def nz(self) -> np.ndarray:
        """Returns cached parent n(z)."""
        self._require_parent()
        return self._parent["nz"]

    def survey_meta(self) -> Mapping[str, Any] | None:
        """Returns cached survey metadata, if available."""
        self._require_parent()
        return self._parent.get("survey_meta")

    def tomo_spec(self) -> Mapping[str, Any]:
        """Returns cached tomography entry spec."""
        self._require_entry()
        return self._last["tomo_spec"]

    def kind(self, default: str = "photoz") -> str:
        """Returns cached kind from tomo_spec, else default."""
        if self._last is None:
            return str(default).strip().lower()
        spec = self._last.get("tomo_spec")
        if not isinstance(spec, Mapping):
            return str(default).strip().lower()
        return str(spec.get("kind", default)).strip().lower()

    def build(
        self,
        *,
        include_metadata: bool = False,
        kind: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ):
        """Build bins using cached parent + cached tomo_spec."""
        self._require_parent()
        self._require_entry()
        self._last, out = _build_bins_from_state(
            parent=self._parent,
            last=self._last,
            include_metadata=include_metadata,
            kind=kind,
            overrides=overrides,
        )
        return out

    def build_from_arrays(
        self,
        *,
        z: Any,
        nz: Any,
        tomo_spec: Mapping[str, Any],
        include_metadata: bool = False,
        survey_meta: Mapping[str, Any] | None = None,
    ):
        """One-shot build from (z, nz, tomo_spec)."""
        self.set_parent_from_arrays(z=z, nz=nz, survey_meta=survey_meta)
        self._last = {
            "tomo_spec": dict(tomo_spec),
            "bins": None,
            "tomo_meta": None,
            "kind": str(tomo_spec.get("kind", "photoz")).strip().lower(),
        }
        return self.build(include_metadata=include_metadata)

    # -------------------------
    # Stats
    # -------------------------
    def bins(self) -> Mapping[int, np.ndarray]:
        """Returns cached bins from the most recent build."""
        self._require_bins()
        return self._last["bins"]

    def tomo_meta(self) -> Mapping[str, Any] | None:
        """Returns cached tomography metadata from the most recent build."""
        self._require_entry()
        return self._last.get("tomo_meta")

    def shape_stats(self, **kwargs) -> dict[str, Any]:
        """Compute shape-only stats for cached bins."""
        self._require_bins()
        return _shape_stats(z=self._parent["z"], bins=self._last["bins"], **kwargs)

    def population_stats(self, **kwargs) -> dict[str, Any]:
        """Compute population stats from cached tomo metadata."""
        self._require_bins()
        meta = self._last.get("tomo_meta")
        if meta is None:
            raise ValueError("No tomo metadata cached. Rebuild with include_metadata=True.")
        return _population_stats(bins=self._last["bins"], metadata=meta, **kwargs)

    # -------------------------
    # Internal
    # -------------------------
    def _require_parent(self) -> None:
        if self._parent is None:
            raise ValueError(
                "No parent (z, nz) cached. Use from_config/from_mapping or set_parent_from_arrays."
            )

    def _require_entry(self) -> None:
        self._require_parent()
        if self._last is None or self._last.get("tomo_spec") is None:
            raise ValueError("No tomo_spec cached. Load an entry or use build_from_arrays.")

    def _require_bins(self) -> None:
        self._require_entry()
        if self._last.get("bins") is None:
            raise ValueError("No bins cached. Call build(...) first.")
