"""Handle for a built tomography (z, nz, bins, metadata).

Defines :class:`TomographyBins`, a lightweight container returned by the
tomography builders. It owns the parent grid, parent n(z), bin curves, and
optional metadata, and provides convenience methods for per-bin statistics,
cross-bin diagnostics, and bin-combination filtering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import binny.nz_tomo.bin_similarity as _bin_sim
from binny.correlations.bin_combo_filter import BinComboFilter
from binny.nz_tomo.bin_stats import population_stats as _population_stats
from binny.nz_tomo.bin_stats import shape_stats as _shape_stats


class TomographyBins:
    """Handle for one built tomography (z, nz, bins, metadata).

    This object owns its data. All statistics and comparisons operate strictly
    on this instance, avoiding any ambiguity from mutable caches elsewhere.
    """

    def __init__(
        self,
        *,
        z: Any,
        nz: Any,
        spec: Mapping[str, Any],
        bins: Mapping[int, Any],
        tomo_meta: Any | None,
        survey_meta: Any | None,
        survey: str | None = None,
    ) -> None:
        self._z = np.asarray(z, dtype=float)
        self._nz = np.asarray(nz, dtype=float)
        self._spec = dict(spec)
        self._bins = {int(k): np.asarray(v, dtype=float) for k, v in bins.items()}
        self._tomo_meta = tomo_meta
        self._survey_meta = survey_meta
        self._survey = None if survey is None else str(survey)

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def nz(self) -> np.ndarray:
        return self._nz

    @property
    def bins(self) -> Mapping[int, np.ndarray]:
        return self._bins

    @property
    def bin_keys(self) -> list[int]:
        return list(self._bins.keys())

    @property
    def spec(self) -> dict[str, Any]:
        return dict(self._spec)

    @property
    def role(self) -> str | None:
        return self._spec.get("role")

    @property
    def year(self) -> Any | None:
        return self._spec.get("year")

    @property
    def survey(self) -> str | None:
        return self._survey

    def shape_stats(self, **kwargs: Any) -> dict[str, Any]:
        return _shape_stats(z=self._z, bins=self._bins, **kwargs)

    def population_stats(self, **kwargs: Any) -> dict[str, Any]:
        if self._tomo_meta is None:
            raise ValueError("No tomo metadata available. Build with include_tomo_metadata=True.")
        return _population_stats(bins=self._bins, metadata=self._tomo_meta, **kwargs)

    def cross_bin_stats(
        self,
        *,
        overlap: Mapping[str, Any] | None = None,
        pairs: Mapping[str, Any] | None = None,
        leakage: Mapping[str, Any] | None = None,
        pearson: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        z = self._z
        bins = self._bins
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

    def make_bin_combo_filter(
        self,
        other: TomographyBins | None = None,
        *,
        curves: Sequence[Mapping[int, Any]] | None = None,
    ) -> BinComboFilter:
        z = self._z
        bins_a = self._bins

        if curves is not None:
            return BinComboFilter(z=z, curves=list(curves))

        if other is None:
            return BinComboFilter(z=z, curves=[bins_a, bins_a])

        if np.asarray(z).shape != np.asarray(other.z).shape or not np.allclose(z, other.z):
            raise ValueError("Combo filter requires a shared z grid.")

        return BinComboFilter(z=z, curves=[bins_a, other.bins])

    def bin_combo_filter(
        self,
        spec: Mapping[str, Any],
        other: TomographyBins | None = None,
    ) -> list[tuple[int, ...]]:
        f = self.make_bin_combo_filter(other)
        return f.select(spec).values()

    def with_survey(self, survey: str) -> TomographyBins:
        return TomographyBins(
            z=self._z,
            nz=self._nz,
            spec=self._spec,
            bins=self._bins,
            tomo_meta=self._tomo_meta,
            survey_meta=self._survey_meta,
            survey=survey,
        )

    @property
    def tomo_meta(self) -> Any | None:
        return self._tomo_meta

    @property
    def survey_meta(self) -> Any | None:
        return self._survey_meta
