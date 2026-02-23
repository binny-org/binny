"""Unit tests for binny.correlations.filters."""

from __future__ import annotations

import pytest

from binny.correlations import filters as flt


def test_apply_comparator_supports_all_tokens():
    """Tests that _apply_comparator supports lt/le/gt/ge."""
    assert flt._apply_comparator(1.0, 2.0, "lt") is True
    assert flt._apply_comparator(2.0, 2.0, "le") is True
    assert flt._apply_comparator(3.0, 2.0, "gt") is True
    assert flt._apply_comparator(2.0, 2.0, "ge") is True
    assert flt._apply_comparator(1.0, 2.0, "ge") is False


def test_apply_comparator_raises_on_unknown_token():
    """Tests that _apply_comparator raises on unknown tokens."""
    with pytest.raises(ValueError, match=r"Unknown comparator"):
        flt._apply_comparator(1.0, 2.0, "nope")


def test_filter_by_score_relation_keeps_tuples_that_pass():
    """Tests that filter_by_score_relation keeps tuples satisfying the relation."""
    tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    scores = [
        {0: 1.0, 1: 10.0},  # pos 0
        {0: 5.0, 1: 6.0},  # pos 1
    ]

    out = flt.filter_by_score_relation(tuples, scores=scores, relation="lt")
    # keep when score[pos1] < score[pos0]
    # (1,0): 5 < 10 -> keep
    # (1,1): 6 < 10 -> keep
    assert out == [(1, 0), (1, 1)]


def test_filter_by_score_relation_raises_on_pos_out_of_range():
    """Tests that filter_by_score_relation raises when a position is out of range."""
    tuples = [(0, 1)]
    scores = [{0: 1.0}, {1: 2.0}]
    with pytest.raises(ValueError, match=r"out of range"):
        flt.filter_by_score_relation(tuples, scores=scores, pos_a=2, pos_b=1)


def test_filter_by_score_relation_raises_on_tuple_too_short():
    """Tests that filter_by_score_relation raises when tuple is too short."""
    tuples = [(0,)]
    scores = [{0: 1.0}, {0: 2.0}]
    with pytest.raises(ValueError, match=r"Tuple length is smaller"):
        flt.filter_by_score_relation(tuples, scores=scores, pos_a=0, pos_b=1)


def test_filter_by_score_relation_raises_on_missing_score():
    """Tests that filter_by_score_relation raises when scores are missing."""
    tuples = [(0, 1)]
    scores = [{0: 1.0}, {}]
    with pytest.raises(KeyError, match=r"Missing score for position 1"):
        flt.filter_by_score_relation(tuples, scores=scores, pos_a=0, pos_b=1)


def test_filter_by_metric_threshold_filters_by_compare_token():
    """Tests that filter_by_metric_threshold filters based on metric(*t) op threshold."""
    tuples = [(0, 0), (0, 2), (2, 2)]

    def metric(i, j):
        return float(i + j)

    out = flt.filter_by_metric_threshold(tuples, metric=metric, threshold=2.0, compare="ge")
    assert out == [(0, 2), (2, 2)]


def test_filter_by_metric_threshold_raises_on_unknown_compare():
    """Tests that filter_by_metric_threshold raises on unknown compare tokens."""
    tuples = [(0, 0)]

    def metric(i, j):
        _, _ = i, j
        return 0.0

    with pytest.raises(ValueError, match=r"Unknown comparator"):
        flt.filter_by_metric_threshold(tuples, metric=metric, threshold=0.0, compare="nope")


def test_filter_by_score_separation_absolute_and_bounds():
    """Tests that filter_by_score_separation supports absolute and bounds logic."""
    tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    scores = [
        {0: 0.0, 1: 10.0},
        {0: 3.0, 1: 12.0},
    ]

    out = flt.filter_by_score_separation(
        tuples,
        scores=scores,
        min_sep=2.0,
        max_sep=5.0,
        absolute=True,
    )
    # separations: (0,0)=3, (0,1)=12, (1,0)=7, (1,1)=2
    assert out == [(0, 0), (1, 1)]


def test_filter_by_score_separation_signed_when_absolute_false():
    """Tests that filter_by_score_separation uses signed separation if absolute=False."""
    tuples = [(0, 0), (0, 1)]
    scores = [{0: 10.0}, {0: 3.0, 1: 12.0}]

    out = flt.filter_by_score_separation(
        tuples,
        scores=scores,
        min_sep=None,
        max_sep=0.0,
        absolute=False,
    )
    # (0,0): 3-10=-7 -> keep (d <= 0)
    # (0,1): 12-10=2  -> reject (d > 0)
    assert out == [(0, 0)]


def test_filter_by_score_separation_rejects_negative_bounds():
    """Tests that filter_by_score_separation rejects negative min/max bounds."""
    tuples = [(0, 0)]
    scores = [{0: 0.0}, {0: 1.0}]
    with pytest.raises(ValueError, match=r"min_sep must be >= 0"):
        flt.filter_by_score_separation(tuples, scores=scores, min_sep=-0.1)
    with pytest.raises(ValueError, match=r"max_sep must be >= 0"):
        flt.filter_by_score_separation(tuples, scores=scores, max_sep=-0.1)


def test_filter_by_score_difference_applies_signed_bounds():
    """Tests that filter_by_score_difference applies signed bounds."""
    tuples = [(0, 0), (0, 1)]
    scores = [{0: 10.0}, {0: 3.0, 1: 12.0}]

    out = flt.filter_by_score_difference(
        tuples,
        scores=scores,
        min_diff=-8.0,
        max_diff=-6.0,
    )
    assert out == [(0, 0)]


def test_filter_by_score_consistency_requires_same_lengths():
    """Tests that filter_by_score_consistency requires scores1/scores2 same length."""
    tuples = [(0, 0)]
    scores1 = [{0: 1.0}, {0: 2.0}]
    scores2 = [{0: 1.0}]
    with pytest.raises(ValueError, match=r"must have the same length"):
        flt.filter_by_score_consistency(tuples, scores1=scores1, scores2=scores2)


def test_filter_by_score_consistency_keeps_only_if_both_pass():
    """Tests that filter_by_score_consistency keeps tuples only if both pass."""
    tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    scores1 = [{0: 1.0, 1: 10.0}, {0: 5.0, 1: 6.0}]
    scores2 = [{0: 2.0, 1: 20.0}, {0: 7.0, 1: 8.0}]

    out = flt.filter_by_score_consistency(
        tuples,
        scores1=scores1,
        scores2=scores2,
        relation="lt",
    )
    # For relation "lt": keep when pos1 < pos0 in both score sets.
    # Only tuples with pos0=1 pass in both sets.
    assert out == [(1, 0), (1, 1)]


def test_filter_by_width_ratio_requires_max_ratio_ge_one():
    """Tests that filter_by_width_ratio rejects max_ratio < 1."""
    tuples = [(0, 0)]
    widths = [{0: 1.0}, {0: 1.0}]
    with pytest.raises(ValueError, match=r"max_ratio must be >= 1"):
        flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=0.9)


def test_filter_by_width_ratio_skips_nonpositive_widths():
    """Tests that filter_by_width_ratio drops tuples with nonpositive widths."""
    tuples = [(0, 0), (0, 1)]
    widths = [{0: 1.0}, {0: 0.0, 1: 2.0}]

    out = flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=10.0)
    # (0,0) skipped due to wb=0, (0,1) kept
    assert out == [(0, 1)]


def test_filter_by_width_ratio_symmetric_behavior():
    """Tests that filter_by_width_ratio applies symmetric ratio when requested."""
    tuples = [(0, 0)]
    widths = [{0: 1.0}, {0: 4.0}]

    out_sym = flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=3.0, symmetric=True)
    out_asym = flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=3.0, symmetric=False)

    # symmetric ratio is max(4, 1/4)=4 -> rejects; asym ratio is 4 -> rejects
    assert out_sym == []
    assert out_asym == []

    out_sym2 = flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=4.0, symmetric=True)
    assert out_sym2 == [(0, 0)]


def test_filter_by_curve_norm_threshold_requires_valid_mode():
    """Tests that filter_by_curve_norm_threshold rejects unknown modes."""
    tuples = [(0, 0)]
    norms = [{0: 1.0}, {0: 1.0}]
    with pytest.raises(ValueError, match=r"Unknown mode"):
        flt.filter_by_curve_norm_threshold(tuples, norms=norms, threshold=0.0, mode="nope")


def test_filter_by_curve_norm_threshold_requires_norms_cover_tuple_length():
    """Tests that filter_by_curve_norm_threshold requires enough norm mappings."""
    tuples = [(0, 0)]
    norms = [{0: 1.0}]
    with pytest.raises(ValueError, match=r"must have at least len\(tuple\)"):
        flt.filter_by_curve_norm_threshold(tuples, norms=norms, threshold=0.0)


def test_filter_by_curve_norm_threshold_raises_on_missing_norm():
    """Tests that filter_by_curve_norm_threshold raises on missing norm entries."""
    tuples = [(0, 1)]
    norms = [{0: 1.0}, {0: 1.0}]
    with pytest.raises(KeyError, match=r"Missing norm for position 1"):
        flt.filter_by_curve_norm_threshold(tuples, norms=norms, threshold=0.0)


def test_filter_by_curve_norm_threshold_all_and_any_modes():
    """Tests that filter_by_curve_norm_threshold supports all/any logic."""
    tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    norms = [{0: 0.1, 1: 10.0}, {0: 0.1, 1: 10.0}]

    out_all = flt.filter_by_curve_norm_threshold(
        tuples,
        norms=norms,
        threshold=1.0,
        compare="ge",
        mode="all",
    )
    out_any = flt.filter_by_curve_norm_threshold(
        tuples,
        norms=norms,
        threshold=1.0,
        compare="ge",
        mode="any",
    )

    # threshold 1: only index=1 passes at each pos.
    assert out_all == [(1, 1)]
    # any: at least one position passes -> tuples with at least one '1'.
    assert out_any == [(0, 1), (1, 0), (1, 1)]


def test_filter_by_width_ratio_skips_when_wa_nonpositive():
    """Tests that filter_by_width_ratio skips tuples with wa <= 0."""
    tuples = [(0, 0), (1, 0)]
    widths = [{0: 0.0, 1: 2.0}, {0: 1.0}]

    out = flt.filter_by_width_ratio(tuples, widths=widths, max_ratio=10.0)
    assert out == [(1, 0)]
