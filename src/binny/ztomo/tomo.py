from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.binny.core.validation import validate_axis_and_weights
from src.binny.axes.edges import mixed_edges
from src.binny.ztomo.photoz import build_photoz_bins
from src.binny.ztomo.specz import build_specz_bins


__all__ = ["TomographicBinner"]


class TomographicBinner:
    """Small helper around (z, nz) to build tomographic bins and edges."""

    def __init__(self, z: ArrayLike, nz: ArrayLike, *, normalized: bool | None = None):
        self.z, self.nz = validate_axis_and_weights(z, nz)

        # track whether nz is already normalized; if not told, infer heuristically
        if normalized is None:
            area = np.trapezoid(self.nz, self.z)
            self.normalized = np.isclose(area, 1.0, rtol=1e-2)
        else:
            self.normalized = bool(normalized)

    # ---------- edges helpers ----------

    def edges_from_mixed(
        self,
        segments: Sequence[Mapping[str, Any]],
        *,
        info_density: ArrayLike | None = None,
        chi: ArrayLike | None = None,
        total_n_bins: int | None = None,
    ) -> np.ndarray:
        """Build bin edges from a sequence of mixed segments."""
        return mixed_edges(
            segments,
            x=self.z,
            weights=self.nz,
            info_density=info_density,
            z=self.z,
            chi=chi,
            total_n_bins=total_n_bins,
        )

    # ---------- build binned n(z) ----------

    def build_photoz_bins(
        self,
        bin_edges: ArrayLike,
        sigma_z_per_bin: Sequence[float],
        z_bias_per_bin: Sequence[float],
        **kwargs: Any,
    ) -> dict[int, np.ndarray]:
        """Forward to src.binny.photoz.build_photoz_bins using this (z, nz)."""
        return build_photoz_bins(
            self.z,
            self.nz,
            bin_edges=bin_edges,
            sigma_z_per_bin=sigma_z_per_bin,
            z_bias_per_bin=z_bias_per_bin,
            **kwargs,
        )

    def build_specz_bins(
        self,
        bin_edges: ArrayLike,
        **kwargs: Any,
    ) -> dict[int, np.ndarray]:
        """Forward to src.binny.specz.build_specz_bins using this (z, nz)."""
        return build_specz_bins(
            self.z,
            self.nz,
            bin_edges=bin_edges,
            **kwargs,
        )

    # ---------- n_eff helpers ----------

    def bin_integrals(self, bins: Mapping[int, np.ndarray]) -> dict[int, float]:
        """∫ n_i(z) dz for each bin."""
        integrals: dict[int, float] = {}
        for idx, nz_bin in bins.items():
            integrals[idx] = float(np.trapezoid(nz_bin, self.z))
        return integrals

    def n_eff_fractions(self, bins: Mapping[int, np.ndarray]) -> dict[int, float]:
        """Fraction of galaxies per bin, assuming parent nz describes total."""
        integrals = self.bin_integrals(bins)
        total = sum(integrals.values())
        if total <= 0:
            raise ValueError("Total integrated n(z) over all bins must be positive.")
        return {idx: val / total for idx, val in integrals.items()}

    def n_eff_per_bin(
        self,
        bins: Mapping[int, np.ndarray],
        n_eff_total: float,
    ) -> dict[int, float]:
        """Map each bin to its n_eff given a total n_eff."""
        frac = self.n_eff_fractions(bins)
        return {idx: n_eff_total * f for idx, f in frac.items()}
