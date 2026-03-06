"""Module for building and filtering bin-index combinations.

This module provides a small, user-facing wrapper for building collections of
index pairs, triplets, or higher-order index tuples and filtering them using
score summaries or curve-based metrics.

The intended use is to pass a shared coordinate grid and per-slot curve
mappings once, then apply high-level filters without manually building
per-position score maps or writing custom tuple metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.utils.types import FloatArray1D, IndexTuple, IndexTuples

from .filters import (
    filter_by_curve_norm_threshold,
    filter_by_metric_threshold,
    filter_by_score_consistency,
    filter_by_score_difference,
    filter_by_score_relation,
    filter_by_score_separation,
    filter_by_width_ratio,
)
from .metrics import (
    metric_from_curves,
    metric_min_overlap_fraction,
    metric_overlap_coefficient,
)
from .scores import (
    score_credible_width,
    score_mean_location,
    score_median_location,
    score_peak_location,
)
from .topology import (
    pairs_all,
    pairs_cartesian,
    pairs_diagonal,
    pairs_lower_triangle,
    pairs_off_diagonal,
    pairs_upper_triangle,
    tuples_all,
    tuples_cartesian,
    tuples_diagonal,
    tuples_nondecreasing,
)

ScoreName = Literal["peak", "mean", "median", "width"]
TopologyName = Literal[
    "pairs_all",
    "pairs_upper_triangle",
    "pairs_lower_triangle",
    "pairs_diagonal",
    "pairs_off_diagonal",
    "pairs_cartesian",
    "tuples_all",
    "tuples_nondecreasing",
    "tuples_diagonal",
    "tuples_cartesian",
]


_SCORE_FNS: dict[str, Callable[..., dict[int, float]]] = {
    "peak": score_peak_location,
    "mean": score_mean_location,
    "median": score_median_location,
    "width": score_credible_width,
}

__all__ = [
    "BinComboFilter",
    "_register_metric_kernel",
    "_available_metric_kernels",
]


METRIC_KERNELS: dict[str, Callable[..., float]] = {}


def _available_metric_kernels() -> list[str]:
    """Returns a sorted list of registered metric kernels."""
    return sorted(METRIC_KERNELS)


def _register_metric_kernel(name: str, func: Callable[..., float]) -> None:
    """Registers a new metric kernel for use with :meth:`BinComboFilter.keep_if_metric`."""
    if name in METRIC_KERNELS:
        raise ValueError(f"Metric kernel {name!r} already registered.")
    METRIC_KERNELS[name] = func


class BinComboFilter:
    """Builds and filters tuples using stored curves.

    The filter stores a shared coordinate grid and one curve mapping per tuple
    position (slot). It also stores the current working list of index tuples.

    The main workflow is:

    - build a tuple collection with :meth:`set_topology`
    - apply one or more filters based on scores or overlap metrics
    - retrieve the resulting tuples with :meth:`values`

    Slot conventions follow tuple positions: slot 0 corresponds to the first
    index in a tuple, slot 1 to the second, and so on.
    """

    def __init__(
        self,
        *,
        z: FloatArray1D,
        curves: Sequence[Mapping[int, FloatArray1D]],
        tuples: Sequence[IndexTuple] | None = None,
    ):
        """Creates a filter for curve-indexed tuple filtering.

        Args:
            z: Shared coordinate grid for all curve evaluations.
            curves: Sequence of per-slot mappings from index to curve values.
                ``curves[p]`` supplies curves for tuple position ``p``.
            tuples: Optional initial tuple collection to filter. If omitted,
                the filter starts with an empty tuple list.
        """
        self.z = z
        self.curves = list(curves)
        tuples_in = tuples or []
        self._tuples: IndexTuples = [tuple(int(x) for x in t) for t in tuples_in]

    def select(self, spec: Mapping[str, Any]) -> BinComboFilter:
        """Apply a YAML-friendly selection spec (topology + ordered filters).

        Informal spec schema::

            spec:
              topology: {name: <TopologyName>, keys?: ...}
              filters:
                - {name: overlap_fraction, threshold: float, compare?: lt/le/gt/ge}
                - {name: overlap_coefficient, threshold: float, compare?: ...}
                - {name: score_relation, score: peak/mean/median/width,
                   pos_a?: int, pos_b?: int, relation?: ...}
                - {name: score_separation, score: ..., min_sep?: float,
                   max_sep?: float, absolute?: bool, ...}
                - {name: score_difference, score: ..., min_diff?: float,
                   max_diff?: float, ...}
                - {name: score_consistency, score1: ..., score2: ...,
                   relation?: ..., ...}
                - {name: width_ratio, max_ratio?: float, symmetric?: bool,
                   pos_a?: int, pos_b?: int, ...}
                - {name: curve_norm_threshold, threshold: float, compare?: ...,
                   mode?: all/any}
                - {name: metric, metric: <kernel_name>, threshold: float,
                   compare?: ...}

        Args:
            spec: Mapping containing an optional topology block and an optional
                ordered list of filter blocks.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If ``spec`` or one of its subentries has the wrong type.
            KeyError: If a requested filter name or metric kernel is unknown.
        """
        # 0) basic validation
        if not isinstance(spec, Mapping):
            raise TypeError("spec must be a mapping.")

        # 1) topology
        topo = spec.get("topology", None)
        if topo is not None and not isinstance(topo, Mapping):
            raise TypeError("spec['topology'] must be a mapping.")

        if topo:
            name = topo["name"]
            keys = topo.get("keys", None)

            if keys is None:
                self.set_topology(name)
            elif isinstance(keys, list) and keys and isinstance(keys[0], list | tuple):
                self.set_topology(name, *keys)
            else:
                self.set_topology(name, keys)

        # 2) filters (ordered)
        fs = spec.get("filters", [])
        if fs is None:
            fs = []
        if not isinstance(fs, Sequence):
            raise TypeError("spec['filters'] must be a sequence.")

        for f in fs:
            if not isinstance(f, Mapping):
                raise TypeError("Each filter entry in spec['filters'] must be a mapping.")

            fname = f["name"]

            match fname:
                case "overlap_fraction":
                    self.keep_if_overlap_fraction(
                        threshold=float(f["threshold"]),
                        compare=f.get("compare", "ge"),
                    )

                case "overlap_coefficient":
                    self.keep_if_overlap_coefficient(
                        threshold=float(f["threshold"]),
                        compare=f.get("compare", "ge"),
                    )

                case "metric":
                    metric_name = f["metric"]
                    if metric_name not in METRIC_KERNELS:
                        raise KeyError(
                            f"Unknown metric kernel {metric_name!r}. "
                            f"Available: {sorted(METRIC_KERNELS)}"
                        )
                    self.keep_if_metric(
                        kernel=METRIC_KERNELS[metric_name],
                        threshold=float(f["threshold"]),
                        compare=f.get("compare", "le"),
                    )

                case "score_relation":
                    self.keep_if_score_relation(
                        score=f["score"],
                        pos_a=int(f.get("pos_a", 0)),
                        pos_b=int(f.get("pos_b", 1)),
                        relation=f.get("relation", "lt"),
                        mass=float(f.get("mass", 0.68)),
                    )

                case "score_separation":
                    self.keep_if_score_separation(
                        score=f["score"],
                        pos_a=int(f.get("pos_a", 0)),
                        pos_b=int(f.get("pos_b", 1)),
                        min_sep=f.get("min_sep", None),
                        max_sep=f.get("max_sep", None),
                        absolute=bool(f.get("absolute", True)),
                        mass=float(f.get("mass", 0.68)),
                    )

                case "score_difference":
                    self.keep_if_score_difference(
                        score=f["score"],
                        pos_a=int(f.get("pos_a", 0)),
                        pos_b=int(f.get("pos_b", 1)),
                        min_diff=f.get("min_diff", None),
                        max_diff=f.get("max_diff", None),
                        mass=float(f.get("mass", 0.68)),
                    )

                case "score_consistency":
                    self.keep_if_score_consistency(
                        score1=f["score1"],
                        score2=f["score2"],
                        pos_a=int(f.get("pos_a", 0)),
                        pos_b=int(f.get("pos_b", 1)),
                        relation=f.get("relation", "lt"),
                        mass=float(f.get("mass", 0.68)),
                        mass2=f.get("mass2", None),
                    )

                case "width_ratio":
                    self.keep_if_width_ratio(
                        pos_a=int(f.get("pos_a", 0)),
                        pos_b=int(f.get("pos_b", 1)),
                        max_ratio=float(f.get("max_ratio", 2.0)),
                        symmetric=bool(f.get("symmetric", True)),
                        mass=float(f.get("mass", 0.68)),
                    )

                case "curve_norm_threshold":
                    self.keep_if_curve_norm_threshold(
                        threshold=float(f["threshold"]),
                        compare=f.get("compare", "ge"),
                        mode=f.get("mode", "all"),
                    )

                case _:
                    raise KeyError(f"Unknown filter name {fname!r}.")

        return self

    def set_topology(self, name: TopologyName, *args, **kwargs) -> BinComboFilter:
        """Builds and stores a tuple collection using a named topology.

        This replaces the current tuple list with the tuples generated by the
        chosen topology.

        Most topology builders require explicit key lists. For convenience, if no
        positional arguments are provided, this method will infer the required key
        inputs from the stored per-slot curve mappings:

        - for within-set pair builders (e.g. "pairs_upper_triangle"), uses the keys
          from slot 0
        - for "pairs_cartesian", uses keys from slots 0 and 1
        - for r-tuple builders that take a single key set (e.g. "tuples_all"),
          uses slot-0 keys and sets r = number of curve slots
        - for "tuples_cartesian", uses keys from every slot

        Args:
            name: Name of a supported topology builder.
            *args: Positional arguments forwarded to the chosen topology. If omitted,
                appropriate defaults are inferred from stored curves as described above.
            **kwargs: Keyword arguments forwarded to the chosen topology. Some defaults
                may be inserted when args are omitted (e.g. r for "tuples_nondecreasing").

        Returns:
            Self, to allow method chaining.

        Raises:
            KeyError: If ``name`` is not a supported topology identifier.
            ValueError: If the chosen topology rejects the provided arguments, or if
                the requested topology cannot be inferred from stored curves (e.g.
                "pairs_cartesian" with fewer than 2 curve slots).
        """
        topo = {
            "pairs_all": pairs_all,
            "pairs_upper_triangle": pairs_upper_triangle,
            "pairs_lower_triangle": pairs_lower_triangle,
            "pairs_diagonal": pairs_diagonal,
            "pairs_off_diagonal": pairs_off_diagonal,
            "pairs_cartesian": pairs_cartesian,
            "tuples_all": tuples_all,
            "tuples_nondecreasing": tuples_nondecreasing,
            "tuples_diagonal": tuples_diagonal,
            "tuples_cartesian": tuples_cartesian,
        }[name]

        if not args:
            match name:
                case (
                    "pairs_all"
                    | "pairs_upper_triangle"
                    | "pairs_lower_triangle"
                    | "pairs_diagonal"
                    | "pairs_off_diagonal"
                ):
                    args = (self._slot_keys(0),)

                case "pairs_cartesian":
                    if len(self.curves) < 2:
                        raise ValueError("pairs_cartesian requires at least 2 curve slots.")
                    args = (self._slot_keys(0), self._slot_keys(1))

                case "tuples_all" | "tuples_diagonal":
                    args = (self._slot_keys(0), len(self.curves))

                case "tuples_nondecreasing":
                    args = (self._slot_keys(0),)
                    kwargs.setdefault("n", len(self.curves))

                case "tuples_cartesian":
                    args = tuple(self._slot_keys(p) for p in range(len(self.curves)))

                case _:
                    # Should be unreachable because TopologyName is constrained,
                    # but keeps things defensive if someone bypasses typing.
                    raise KeyError(f"Unknown topology {name!r}.")

        self._tuples = topo(*args, **kwargs)
        return self

    def values(self) -> IndexTuples:
        """Returns the current filtered tuple list.

        Returns:
            A list of index tuples representing the current filtered result.
        """
        return list(self._tuples)

    def _scores(self, score: ScoreName, *, mass: float = 0.68) -> list[dict[int, float]]:
        """Computes per-slot score maps from the stored curves.

        Args:
            score: Name of the score definition to use for each curve.
            mass: Credible mass used when ``score="width"``.

        Returns:
            A list of mappings, one per slot, where each mapping assigns a
            scalar score to each available index in that slot.
        """
        out: list[dict[int, float]] = []
        for slot_curves in self.curves:
            fn = _SCORE_FNS[score]
            if score == "width":
                out.append(fn(z=self.z, curves=slot_curves, mass=mass))  # type: ignore[misc]
            else:
                out.append(fn(z=self.z, curves=slot_curves))  # type: ignore[misc]
        return out

    def keep_if_score_relation(
        self,
        score: ScoreName,
        *,
        pos_a: int = 0,
        pos_b: int = 1,
        relation: Literal["lt", "le", "gt", "ge"] = "lt",
        mass: float = 0.68,
    ) -> BinComboFilter:
        """Keeps tuples based on an ordering between two score values.

        For each tuple, computes the requested score at positions ``pos_a`` and
        ``pos_b`` and keeps the tuple when the score at ``pos_b`` satisfies the
        requested relation relative to the score at ``pos_a``.

        Args:
            score: Score definition used to reduce each curve to a scalar.
            pos_a: First tuple position in the comparison.
            pos_b: Second tuple position in the comparison.
            relation: Comparison operator applied to the two score values.
            mass: Credible mass used when ``score="width"``.

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If requested positions are not valid for the tuple
                collection.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        scores = self._scores(score, mass=mass)
        self._tuples = filter_by_score_relation(
            self._tuples, scores=scores, pos_a=pos_a, pos_b=pos_b, relation=relation
        )
        return self

    def keep_if_score_separation(
        self,
        score: ScoreName,
        *,
        pos_a: int = 0,
        pos_b: int = 1,
        min_sep: float | None = None,
        max_sep: float | None = None,
        absolute: bool = True,
        mass: float = 0.68,
    ) -> BinComboFilter:
        """Keeps tuples whose score separation lies within a requested window.

        For each tuple, computes the score difference between positions
        ``pos_a`` and ``pos_b`` and keeps the tuple when the separation lies
        within the provided bounds (ignoring bounds set to ``None``).

        Args:
            score: Score definition used to reduce each curve to a scalar.
            pos_a: First tuple position used in the separation.
            pos_b: Second tuple position used in the separation.
            min_sep: Optional minimum separation to enforce.
            max_sep: Optional maximum separation to enforce.
            absolute: If True, apply bounds to the absolute separation.
            mass: Credible mass used when ``score="width"``.

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If requested positions are invalid or bounds are
                inconsistent.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        scores = self._scores(score, mass=mass)
        self._tuples = filter_by_score_separation(
            self._tuples,
            scores=scores,
            pos_a=pos_a,
            pos_b=pos_b,
            min_sep=min_sep,
            max_sep=max_sep,
            absolute=absolute,
        )
        return self

    def keep_if_width_ratio(
        self,
        *,
        pos_a: int = 0,
        pos_b: int = 1,
        max_ratio: float = 2.0,
        symmetric: bool = True,
        mass: float = 0.68,
    ) -> BinComboFilter:
        """Keeps tuples whose width values are compatible across two positions.

        This filter compares per-index width-like scores between two tuple
        positions and keeps tuples where the ratio does not exceed
        ``max_ratio``. When ``symmetric=True``, the ratio is treated
        symmetrically between the two positions.

        Args:
            pos_a: First tuple position in the comparison.
            pos_b: Second tuple position in the comparison.
            max_ratio: Maximum allowed ratio between widths.
            symmetric: If True, enforce ratio symmetry between positions.
            mass: Credible mass used to define widths.

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If requested positions are invalid or ``max_ratio`` is
                not meaningful.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        widths = self._scores("width", mass=mass)
        self._tuples = filter_by_width_ratio(
            self._tuples,
            widths=widths,
            pos_a=pos_a,
            pos_b=pos_b,
            max_ratio=max_ratio,
            symmetric=symmetric,
        )
        return self

    def keep_if_overlap_fraction(
        self,
        *,
        threshold: float,
        compare: Literal["lt", "le", "gt", "ge"] = "ge",
    ) -> BinComboFilter:
        """Keeps tuples based on the minimum-overlap fraction.

        This filter computes the normalized minimum-overlap fraction for the
        curves selected by each tuple and keeps tuples that satisfy the chosen
        comparison against ``threshold``.

        Args:
            threshold: Reference value used for filtering.
            compare: Comparison operator applied as value op threshold.

        Returns:
            Self, to allow method chaining.
        """
        metric = metric_min_overlap_fraction(z=self.z, curves=self.curves)
        self._tuples = filter_by_metric_threshold(
            self._tuples, metric=metric, threshold=threshold, compare=compare
        )
        return self

    def keep_if_overlap_coefficient(
        self,
        *,
        threshold: float,
        compare: Literal["lt", "le", "gt", "ge"] = "ge",
    ) -> BinComboFilter:
        """Keeps tuples based on the overlap coefficient.

        This filter computes the overlap coefficient for the curves selected by
        each tuple and keeps tuples that satisfy the chosen comparison against
        ``threshold``.

        Args:
            threshold: Reference value used for filtering.
            compare: Comparison operator applied as value op threshold.

        Returns:
            Self, to allow method chaining.
        """
        metric = metric_overlap_coefficient(z=self.z, curves=self.curves)
        self._tuples = filter_by_metric_threshold(
            self._tuples, metric=metric, threshold=threshold, compare=compare
        )
        return self

    def keep_if_metric(
        self,
        *,
        kernel: Callable[..., float],
        threshold: float,
        compare: Literal["lt", "le", "gt", "ge"] = "le",
    ) -> BinComboFilter:
        """Keeps tuples based on a user-supplied curve metric.

        The provided kernel is treated as a curve-level metric evaluated on the
        curves selected by each tuple. Tuples are kept when the resulting value
        satisfies the chosen comparison against ``threshold``.

        Args:
            kernel: Callable that maps N curves to a scalar metric value.
            threshold: Reference value used for filtering.
            compare: Comparison operator applied as value op threshold.

        Returns:
            Self, to allow method chaining.
        """
        metric = metric_from_curves(curves=self.curves, kernel=kernel)
        self._tuples = filter_by_metric_threshold(
            self._tuples, metric=metric, threshold=threshold, compare=compare
        )
        return self

    def keep_if_score_difference(
        self,
        score: ScoreName,
        *,
        pos_a: int = 0,
        pos_b: int = 1,
        min_diff: float | None = None,
        max_diff: float | None = None,
        mass: float = 0.68,
    ) -> BinComboFilter:
        """Keeps tuples whose signed score difference lies within a window.

        For each tuple, computes the signed score difference between positions
        ``pos_a`` and ``pos_b`` and keeps the tuple when the difference lies
        within the provided bounds (ignoring bounds set to ``None``).

        This supports directional selections such as keeping tuples where the
        score at one position is larger than the score at another.

        Args:
            score: Score definition used to reduce each curve to a scalar.
            pos_a: First tuple position used in the difference.
            pos_b: Second tuple position used in the difference.
            min_diff: Optional minimum signed difference to enforce.
            max_diff: Optional maximum signed difference to enforce.
            mass: Credible mass used when ``score="width"``.

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If requested positions are not valid for the tuple
                collection.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        scores = self._scores(score, mass=mass)
        self._tuples = filter_by_score_difference(
            self._tuples,
            scores=scores,
            pos_a=pos_a,
            pos_b=pos_b,
            min_diff=min_diff,
            max_diff=max_diff,
        )
        return self

    def keep_if_score_consistency(
        self,
        score1: ScoreName,
        score2: ScoreName,
        *,
        pos_a: int = 0,
        pos_b: int = 1,
        relation: Literal["lt", "le", "gt", "ge"] = "lt",
        mass: float = 0.68,
        mass2: float | None = None,
    ) -> BinComboFilter:
        """Keeps tuples that satisfy the same ordering under two score choices.

        For each tuple, evaluates two different score definitions and keeps the
        tuple only if the requested ordering between ``pos_a`` and ``pos_b``
        holds for both scores.

        This is useful when a robust ordering that is stable under
        alternative summaries is required, such as peak and mean locations.

        Args:
            score1: First score definition used to summarize each curve.
            score2: Second score definition used to summarize each curve.
            pos_a: First tuple position in the comparison.
            pos_b: Second tuple position in the comparison.
            relation: Comparison operator applied for both score definitions.
            mass: Credible mass used when ``score1="width"``.
            mass2: Credible mass used when ``score2="width"``. If None, uses
                ``mass``.

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If requested positions are not valid for the tuple
                collection.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        scores1 = self._scores(score1, mass=mass)
        scores2 = self._scores(score2, mass=mass if mass2 is None else mass2)
        self._tuples = filter_by_score_consistency(
            self._tuples,
            scores1=scores1,
            scores2=scores2,
            pos_a=pos_a,
            pos_b=pos_b,
            relation=relation,
        )
        return self

    def keep_if_curve_norm_threshold(
        self,
        *,
        threshold: float,
        compare: Literal["lt", "le", "gt", "ge"] = "ge",
        mode: Literal["all", "any"] = "all",
    ) -> BinComboFilter:
        """Keeps tuples based on per-slot curve normalization thresholds.

        This filter uses the integrated curve norm at each tuple position to
        exclude tuples that involve curves with very small integrated area.

        With ``mode="all"``, every position must pass the threshold. With
        ``mode="any"``, at least one position must pass.

        Args:
            threshold: Reference value used for filtering.
            compare: Comparison operator applied to each norm versus threshold.
            mode: Whether to require all positions ("all") or at least one
                ("any").

        Returns:
            Self, to allow method chaining.

        Raises:
            ValueError: If the mode is not recognized, or norms do not cover the
                tuple positions.
            KeyError: If a required index is not present in the stored curves
                for a slot.
        """
        zz = np.asarray(self.z, dtype=float)
        norm_maps: list[dict[int, float]] = []
        for slot_curves in self.curves:
            slot_norms: dict[int, float] = {}
            for k, c in slot_curves.items():
                cc = np.asarray(c, dtype=float)
                slot_norms[int(k)] = float(np.trapezoid(cc, zz))
            norm_maps.append(slot_norms)

        self._tuples = filter_by_curve_norm_threshold(
            self._tuples,
            norms=norm_maps,
            threshold=threshold,
            compare=compare,
            mode=mode,
        )
        return self

    def _slot_keys(self, slot: int) -> list[int]:
        """Returns the available index keys for a curve slot.

        Keys are returned in insertion order (the mapping order of
        ``self.curves[slot]``). This is used to auto-fill topology arguments
        when the user does not provide explicit key lists.

        Args:
            slot: Slot index (tuple position) whose curve-key set is returned.

        Returns:
            A list of integer keys present in the requested slot mapping.
        """
        # Slot keys in insertion order (dict preserves insertion order).
        return list(self.curves[slot].keys())
