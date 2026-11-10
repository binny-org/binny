"""Correlation topology, metrics, scores, and filtering."""

from binny.correlations.bin_combo_filter import BinComboFilter
from binny.correlations.filters import (
    filter_by_curve_norm_threshold,
    filter_by_metric_threshold,
    filter_by_score_consistency,
    filter_by_score_difference,
    filter_by_score_relation,
    filter_by_score_separation,
    filter_by_width_ratio,
)
from binny.correlations.metrics import (
    metric_from_curves,
    metric_min_overlap_fraction,
    metric_overlap_coefficient,
)
from binny.correlations.scores import (
    score_credible_width,
    score_mean_location,
    score_median_location,
    score_peak_location,
)
from binny.correlations.topology import (
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

__all__ = [
    "BinComboFilter",
    "filter_by_curve_norm_threshold",
    "filter_by_metric_threshold",
    "filter_by_score_consistency",
    "filter_by_score_difference",
    "filter_by_score_relation",
    "filter_by_score_separation",
    "filter_by_width_ratio",
    "metric_from_curves",
    "metric_min_overlap_fraction",
    "metric_overlap_coefficient",
    "score_credible_width",
    "score_mean_location",
    "score_median_location",
    "score_peak_location",
    "pairs_all",
    "pairs_cartesian",
    "pairs_diagonal",
    "pairs_lower_triangle",
    "pairs_off_diagonal",
    "pairs_upper_triangle",
    "tuples_all",
    "tuples_cartesian",
    "tuples_diagonal",
    "tuples_nondecreasing",
]
