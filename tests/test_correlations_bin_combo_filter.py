"""Unit tests for binny.correlations.bin_combo_filter."""

from __future__ import annotations

import numpy as np
import pytest

import binny.correlations.bin_combo_filter as bcf


def _toy_z(n: int = 11) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _toy_curves(z: np.ndarray) -> list[dict[int, np.ndarray]]:
    # Slot 0: indices 0,1
    # Slot 1: indices 0,1
    c00 = np.ones_like(z) * 1.0
    c01 = np.ones_like(z) * 2.0
    c10 = np.ones_like(z) * 3.0
    c11 = np.ones_like(z) * 4.0
    return [{0: c00, 1: c01}, {0: c10, 1: c11}]


def test_available_metric_kernels_sorted_and_empty_by_default(monkeypatch):
    """Tests that _available_metric_kernels returns sorted kernel names."""

    def k_b(*_args: object, **_kwargs: object) -> float:
        return 0.0

    def k_a(*_args: object, **_kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(bcf, "METRIC_KERNELS", {"b": k_b, "a": k_a})
    assert bcf._available_metric_kernels() == ["a", "b"]


def test_register_metric_kernel_adds_and_rejects_duplicates(monkeypatch):
    """Tests that _register_metric_kernel registers kernels and rejects duplicates."""

    def k0(*_args: object, **_kwargs: object) -> float:
        return 0.0

    def k0_dupe(*_args: object, **_kwargs: object) -> float:
        return 1.0

    bcf._register_metric_kernel("k0", k0)
    assert "k0" in bcf.METRIC_KERNELS

    with pytest.raises(ValueError, match=r"already registered"):
        bcf._register_metric_kernel("k0", k0_dupe)


def test_init_casts_tuple_entries_to_ints():
    """Tests that BinComboFilter casts tuple elements to ints."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1.2), (1.9, 0)])
    assert f.values() == [(0, 1), (1, 0)]


def test_values_returns_copy_of_current_tuples():
    """Tests that values returns a list of the current tuples."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])
    out = f.values()
    assert out == [(0, 1)]
    assert out is not f._tuples


def test_slot_keys_preserves_insertion_order():
    """Tests that _slot_keys returns keys in mapping insertion order."""
    z = _toy_z()
    curves = [{5: np.ones_like(z), 2: np.ones_like(z)}, {9: np.ones_like(z)}]
    f = bcf.BinComboFilter(z=z, curves=curves)
    assert f._slot_keys(0) == [5, 2]
    assert f._slot_keys(1) == [9]


def test_scores_calls_correct_score_functions_and_passes_mass(monkeypatch):
    """Tests that _scores dispatches to the right score function and passes mass."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    called = {"peak": 0, "width": 0, "mass": None}

    def fake_peak(*, z, curves):
        _ = z
        called["peak"] += 1
        return {k: float(k) for k in curves}

    def fake_width(*, z, curves, mass):
        _ = z
        called["width"] += 1
        called["mass"] = mass
        return {k: 10.0 + float(k) for k in curves}

    monkeypatch.setattr(bcf, "_SCORE_FNS", {"peak": fake_peak, "width": fake_width})

    out_peak = f._scores("peak")
    assert called["peak"] == 2
    assert isinstance(out_peak, list) and len(out_peak) == 2
    assert out_peak[0][0] == 0.0

    out_width = f._scores("width", mass=0.9)
    assert called["width"] == 2
    assert called["mass"] == 0.9
    assert out_width[0][0] == 10.0


def test_set_topology_infers_pair_topology_keys_from_slot0(monkeypatch):
    """Tests that set_topology infers within-set pair keys from slot 0."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"args": None, "kwargs": None}

    def fake_pairs_upper_triangle(keys):
        got["args"] = (list(keys),)
        got["kwargs"] = {}
        return [(0, 1)]

    monkeypatch.setattr(bcf, "pairs_upper_triangle", fake_pairs_upper_triangle)

    f.set_topology("pairs_upper_triangle")
    assert got["args"] == ([0, 1],)
    assert f.values() == [(0, 1)]


def test_set_topology_pairs_cartesian_requires_two_slots(monkeypatch):
    """Tests that pairs_cartesian inference requires at least two curve slots."""
    z = _toy_z()
    curves = [{0: np.ones_like(z)}]
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(ValueError, match=r"requires at least 2 curve slots"):
        f.set_topology("pairs_cartesian")


def test_set_topology_infers_pairs_cartesian_keys_from_slots_0_1(monkeypatch):
    """Tests that set_topology infers pairs_cartesian keys from slots 0 and 1."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"args": None}

    def fake_pairs_cartesian(keys0, keys1):
        got["args"] = (list(keys0), list(keys1))
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

    monkeypatch.setattr(bcf, "pairs_cartesian", fake_pairs_cartesian)

    f.set_topology("pairs_cartesian")
    assert got["args"] == ([0, 1], [0, 1])
    assert len(f.values()) == 4


def test_set_topology_infers_tuples_all_and_diagonal(monkeypatch):
    """Tests that tuples_all/tuples_diagonal infer (keys, r) from curves."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"all": None, "diag": None}

    def fake_tuples_all(keys, r):
        got["all"] = (list(keys), int(r))
        return [(0, 0), (1, 1)]

    def fake_tuples_diagonal(keys, r):
        got["diag"] = (list(keys), int(r))
        return [(0, 0)]

    monkeypatch.setattr(bcf, "tuples_all", fake_tuples_all)
    monkeypatch.setattr(bcf, "tuples_diagonal", fake_tuples_diagonal)

    f.set_topology("tuples_all")
    assert got["all"] == ([0, 1], 2)

    f.set_topology("tuples_diagonal")
    assert got["diag"] == ([0, 1], 2)


def test_set_topology_infers_tuples_nondecreasing_sets_n_kw(monkeypatch):
    """Tests that tuples_nondecreasing inference sets n=len(curves) kwarg."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"args": None, "kwargs": None}

    def fake_tuples_nondecreasing(keys, *, n):
        got["args"] = (list(keys),)
        got["kwargs"] = {"n": n}
        return [(0, 1)]

    monkeypatch.setattr(bcf, "tuples_nondecreasing", fake_tuples_nondecreasing)

    f.set_topology("tuples_nondecreasing")
    assert got["args"] == ([0, 1],)
    assert got["kwargs"]["n"] == 2


def test_set_topology_infers_tuples_cartesian_from_all_slots(monkeypatch):
    """Tests that tuples_cartesian inference uses keys from every slot."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"args": None}

    def fake_tuples_cartesian(keys0, keys1):
        got["args"] = (list(keys0), list(keys1))
        return [(0, 0), (1, 1)]

    monkeypatch.setattr(bcf, "tuples_cartesian", fake_tuples_cartesian)

    f.set_topology("tuples_cartesian")
    assert got["args"] == ([0, 1], [0, 1])


def test_select_rejects_non_mapping_spec():
    """Tests that select rejects non-mapping specs."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(TypeError, match=r"spec must be a mapping"):
        f.select(["nope"])  # type: ignore[arg-type]


def test_select_rejects_non_mapping_topology():
    """Tests that select rejects a non-mapping topology value."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(TypeError, match=r"spec\['topology'\] must be a mapping"):
        f.select({"topology": "pairs_all"})


def test_select_rejects_non_sequence_filters():
    """Tests that select rejects non-sequence filters values."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(TypeError, match=r"spec\['filters'\] must be a sequence"):
        f.select({"filters": {"name": "overlap_fraction"}})


def test_select_rejects_non_mapping_filter_entries():
    """Tests that select rejects non-mapping filter entries."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(TypeError, match=r"Each filter entry"):
        f.select({"filters": ["nope"]})


def test_select_applies_topology_and_dispatches_filters_in_order(monkeypatch):
    """Tests that select dispatches filters in order with parsed arguments."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    order: list[str] = []

    monkeypatch.setattr(
        bcf.BinComboFilter,
        "set_topology",
        lambda self, name, *a: order.append(f"topo:{name}") or self,
        raising=True,
    )
    monkeypatch.setattr(
        bcf.BinComboFilter,
        "keep_if_overlap_fraction",
        lambda self, **k: order.append("overlap_fraction") or self,
        raising=True,
    )
    monkeypatch.setattr(
        bcf.BinComboFilter,
        "keep_if_score_relation",
        lambda self, score, **k: order.append(f"score_relation:{score}") or self,
        raising=True,
    )
    monkeypatch.setattr(
        bcf.BinComboFilter,
        "keep_if_curve_norm_threshold",
        lambda self, **k: order.append("curve_norm_threshold") or self,
        raising=True,
    )

    spec = {
        "topology": {"name": "pairs_all"},
        "filters": [
            {"name": "overlap_fraction", "threshold": 0.1, "compare": "ge"},
            {"name": "score_relation", "score": "mean", "relation": "lt"},
            {"name": "curve_norm_threshold", "threshold": 0.0, "mode": "all"},
        ],
    }
    out = f.select(spec)
    assert out is f
    assert order == [
        "topo:pairs_all",
        "overlap_fraction",
        "score_relation:mean",
        "curve_norm_threshold",
    ]


def test_select_raises_on_unknown_filter_name():
    """Tests that select raises KeyError on unknown filter names."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(KeyError, match=r"Unknown filter name"):
        f.select({"filters": [{"name": "wat"}]})


def test_select_metric_raises_on_unknown_kernel(monkeypatch):
    """Tests that select raises on unknown metric kernel."""
    monkeypatch.setattr(bcf, "METRIC_KERNELS", {})

    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(KeyError, match=r"Unknown metric kernel"):
        f.select({"filters": [{"name": "metric", "metric": "nope", "threshold": 1.0}]})


def test_select_metric_dispatches_keep_if_metric_with_registered_kernel(monkeypatch):
    """Tests that select dispatches metric filters to keep_if_metric."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    def k(*curves) -> float:
        _ = curves
        return 0.0

    monkeypatch.setattr(bcf, "METRIC_KERNELS", {"k": k})

    called = {"kernel": None, "threshold": None, "compare": None}

    def fake_keep_if_metric(self, *, kernel, threshold, compare="le"):
        called["kernel"] = kernel
        called["threshold"] = threshold
        called["compare"] = compare
        return self

    monkeypatch.setattr(bcf.BinComboFilter, "keep_if_metric", fake_keep_if_metric, raising=True)

    f.select({"filters": [{"name": "metric", "metric": "k", "threshold": 2.0}]})
    assert called["kernel"] is k
    assert called["threshold"] == 2.0
    assert called["compare"] == "le"


def test_keep_if_overlap_fraction_delegates_to_metric_and_filter(monkeypatch):
    """Tests that keep_if_overlap_fraction delegates to metric + threshold filter."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    fake_metric = object()

    def fake_metric_min_overlap_fraction(*, z, curves):
        _, _ = z, curves
        return fake_metric

    got = {"metric": None, "threshold": None, "compare": None}

    def fake_filter_by_metric_threshold(tuples, *, metric, threshold, compare):
        got["metric"] = metric
        got["threshold"] = threshold
        got["compare"] = compare
        return tuples

    monkeypatch.setattr(bcf, "metric_min_overlap_fraction", fake_metric_min_overlap_fraction)
    monkeypatch.setattr(bcf, "filter_by_metric_threshold", fake_filter_by_metric_threshold)

    f.keep_if_overlap_fraction(threshold=0.5, compare="ge")
    assert got["metric"] is fake_metric
    assert got["threshold"] == 0.5
    assert got["compare"] == "ge"


def test_keep_if_overlap_coefficient_delegates_to_metric_and_filter(monkeypatch):
    """Tests that keep_if_overlap_coefficient delegates to metric + threshold filter."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    fake_metric = object()

    def fake_metric_overlap_coefficient(*, z, curves):
        _, _ = z, curves
        return fake_metric

    got = {"metric": None}

    def fake_filter_by_metric_threshold(tuples, *, metric, threshold, compare):
        _ = threshold, compare
        got["metric"] = metric
        return tuples

    monkeypatch.setattr(bcf, "metric_overlap_coefficient", fake_metric_overlap_coefficient)
    monkeypatch.setattr(bcf, "filter_by_metric_threshold", fake_filter_by_metric_threshold)

    f.keep_if_overlap_coefficient(threshold=0.1, compare="lt")
    assert got["metric"] is fake_metric


def test_keep_if_metric_delegates_to_metric_from_curves_and_filter(monkeypatch):
    """Tests that keep_if_metric delegates to metric_from_curves and threshold filter."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    def kernel(*_args: object, **_kwargs: object) -> float:
        """Return a constant zero value regardless of inputs."""
        return 0.0

    fake_metric = object()
    got = {"kernel": None, "metric": None}

    def fake_metric_from_curves(*, curves, kernel):
        _ = curves
        got["kernel"] = kernel
        return fake_metric

    def fake_filter_by_metric_threshold(tuples, *, metric, threshold, compare):
        got["metric"] = metric
        return tuples

    monkeypatch.setattr(bcf, "metric_from_curves", fake_metric_from_curves)
    monkeypatch.setattr(bcf, "filter_by_metric_threshold", fake_filter_by_metric_threshold)

    f.keep_if_metric(kernel=kernel, threshold=1.0, compare="le")
    assert got["kernel"] is kernel
    assert got["metric"] is fake_metric


def test_keep_if_score_relation_delegates(monkeypatch):
    """Tests that keep_if_score_relation delegates to filter_by_score_relation."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    monkeypatch.setattr(bcf.BinComboFilter, "_scores", lambda self, score, mass=0.68: ["S"])

    got = {"scores": None, "pos_a": None, "pos_b": None, "relation": None}

    def fake_filter(tuples, *, scores, pos_a, pos_b, relation):
        got["scores"] = scores
        got["pos_a"] = pos_a
        got["pos_b"] = pos_b
        got["relation"] = relation
        return tuples

    monkeypatch.setattr(bcf, "filter_by_score_relation", fake_filter)

    f.keep_if_score_relation("mean", pos_a=0, pos_b=1, relation="lt")
    assert got["scores"] == ["S"]
    assert got["pos_a"] == 0
    assert got["pos_b"] == 1
    assert got["relation"] == "lt"


def test_keep_if_score_separation_delegates(monkeypatch):
    """Tests that keep_if_score_separation delegates to filter_by_score_separation."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    monkeypatch.setattr(bcf.BinComboFilter, "_scores", lambda self, score, mass=0.68: ["S"])

    got = {"min_sep": None, "max_sep": None, "absolute": None}

    def fake_filter(tuples, *, scores, pos_a, pos_b, min_sep, max_sep, absolute):
        got["min_sep"] = min_sep
        got["max_sep"] = max_sep
        got["absolute"] = absolute
        return tuples

    monkeypatch.setattr(bcf, "filter_by_score_separation", fake_filter)

    f.keep_if_score_separation("median", min_sep=0.1, max_sep=0.5, absolute=False)
    assert got["min_sep"] == 0.1
    assert got["max_sep"] == 0.5
    assert got["absolute"] is False


def test_keep_if_score_difference_delegates(monkeypatch):
    """Tests that keep_if_score_difference delegates to filter_by_score_difference."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    monkeypatch.setattr(bcf.BinComboFilter, "_scores", lambda self, score, mass=0.68: ["S"])

    got = {"min_diff": None, "max_diff": None}

    def fake_filter(tuples, *, scores, pos_a, pos_b, min_diff, max_diff):
        got["min_diff"] = min_diff
        got["max_diff"] = max_diff
        return tuples

    monkeypatch.setattr(bcf, "filter_by_score_difference", fake_filter)

    f.keep_if_score_difference("peak", min_diff=-1.0, max_diff=2.0)
    assert got["min_diff"] == -1.0
    assert got["max_diff"] == 2.0


def test_keep_if_score_consistency_delegates(monkeypatch):
    """Tests that keep_if_score_consistency delegates to filter_by_score_consistency."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    def fake_scores(self, score, mass=0.68):
        return [f"{score}:{mass}"]

    monkeypatch.setattr(bcf.BinComboFilter, "_scores", fake_scores)

    got = {"scores1": None, "scores2": None, "relation": None}

    def fake_filter(tuples, *, scores1, scores2, pos_a, pos_b, relation):
        got["scores1"] = scores1
        got["scores2"] = scores2
        got["relation"] = relation
        return tuples

    monkeypatch.setattr(bcf, "filter_by_score_consistency", fake_filter)

    f.keep_if_score_consistency(
        "width",
        "mean",
        relation="ge",
        mass=0.7,
        mass2=0.9,
    )
    assert got["scores1"] == ["width:0.7"]
    assert got["scores2"] == ["mean:0.9"]
    assert got["relation"] == "ge"


def test_keep_if_width_ratio_delegates(monkeypatch):
    """Tests that keep_if_width_ratio delegates to filter_by_width_ratio."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    monkeypatch.setattr(bcf.BinComboFilter, "_scores", lambda self, score, mass=0.68: ["W"])

    got = {"max_ratio": None, "symmetric": None}

    def fake_filter(tuples, *, widths, pos_a, pos_b, max_ratio, symmetric):
        got["max_ratio"] = max_ratio
        got["symmetric"] = symmetric
        return tuples

    monkeypatch.setattr(bcf, "filter_by_width_ratio", fake_filter)

    f.keep_if_width_ratio(max_ratio=1.5, symmetric=False)
    assert got["max_ratio"] == 1.5
    assert got["symmetric"] is False


def test_keep_if_curve_norm_threshold_computes_norms_and_delegates(monkeypatch):
    """Tests that keep_if_curve_norm_threshold computes per-slot norms and delegates."""
    z = np.array([0.0, 1.0], dtype=float)
    c0 = np.array([1.0, 1.0], dtype=float)  # trapezoid -> 1.0
    c1 = np.array([2.0, 2.0], dtype=float)  # trapezoid -> 2.0
    curves = [{0: c0}, {1: c1}]
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    got = {"norms": None, "threshold": None, "compare": None, "mode": None}

    def fake_filter(tuples, *, norms, threshold, compare, mode):
        got["norms"] = norms
        got["threshold"] = threshold
        got["compare"] = compare
        got["mode"] = mode
        return tuples

    monkeypatch.setattr(bcf, "filter_by_curve_norm_threshold", fake_filter)

    f.keep_if_curve_norm_threshold(threshold=0.5, compare="ge", mode="all")
    assert got["threshold"] == 0.5
    assert got["compare"] == "ge"
    assert got["mode"] == "all"
    assert got["norms"][0][0] == pytest.approx(1.0)
    assert got["norms"][1][1] == pytest.approx(2.0)


def test_set_topology_raises_on_unknown_topology_name():
    """Tests that set_topology raises KeyError on unknown topology names."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(KeyError, match=r"pairs_weird"):
        f.set_topology("pairs_weird")  # type: ignore[arg-type]


def test_select_raises_on_unknown_topology_name():
    """Tests that select raises KeyError when spec requests an unknown topology."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(KeyError, match=r"pairs_weird"):
        f.select({"topology": {"name": "pairs_weird"}})


def test_select_passes_nested_keys_as_star_args_to_topology(monkeypatch):
    """Tests that select expands a nested keys list into positional topology args."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"name": None, "args": None}

    def fake_set_topology(self, name, *args):
        got["name"] = name
        got["args"] = args
        return self

    monkeypatch.setattr(bcf.BinComboFilter, "set_topology", fake_set_topology, raising=True)

    f.select(
        {
            "topology": {
                "name": "pairs_cartesian",
                "keys": [[0, 1], [10, 11]],
            }
        }
    )

    assert got["name"] == "pairs_cartesian"
    assert got["args"] == ([0, 1], [10, 11])


def test_select_passes_flat_keys_as_single_topology_argument(monkeypatch):
    """Tests that select passes a flat keys list as one topology argument."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    got = {"name": None, "args": None}

    def fake_set_topology(self, name, *args):
        got["name"] = name
        got["args"] = args
        return self

    monkeypatch.setattr(bcf.BinComboFilter, "set_topology", fake_set_topology, raising=True)

    f.select(
        {
            "topology": {
                "name": "pairs_upper_triangle",
                "keys": [3, 4, 5],
            }
        }
    )

    assert got["name"] == "pairs_upper_triangle"
    assert got["args"] == ([3, 4, 5],)


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("keep_if_overlap_fraction", {"threshold": 0.5, "compare": "bad"}),
        ("keep_if_overlap_coefficient", {"threshold": 0.5, "compare": "bad"}),
        ("keep_if_metric", {"kernel": lambda *_args: 0.0, "threshold": 0.5, "compare": "bad"}),
        ("keep_if_score_relation", {"score": "mean", "relation": "bad"}),
        ("keep_if_score_consistency", {"score1": "mean", "score2": "median", "relation": "bad"}),
        ("keep_if_curve_norm_threshold", {"threshold": 0.5, "compare": "bad", "mode": "all"}),
    ],
)
def test_methods_raise_on_invalid_relation_strings(method_name, kwargs):
    """Tests that public methods reject unsupported relation identifiers."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves, tuples=[(0, 1)])

    method = getattr(f, method_name)

    with pytest.raises(ValueError, match=r"Unknown relation"):
        method(**kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        {"filters": [{"name": "overlap_fraction", "threshold": 0.1, "compare": "bad"}]},
        {"filters": [{"name": "overlap_coefficient", "threshold": 0.1, "compare": "bad"}]},
        {"filters": [{"name": "score_relation", "score": "mean", "relation": "bad"}]},
        {
            "filters": [
                {
                    "name": "score_consistency",
                    "score1": "mean",
                    "score2": "median",
                    "relation": "bad",
                }
            ]
        },
        {"filters": [{"name": "curve_norm_threshold", "threshold": 0.0, "compare": "bad"}]},
    ],
)
def test_select_raises_on_invalid_relation_strings(spec):
    """Tests that select rejects unsupported relation identifiers in filter specs."""
    z = _toy_z()
    curves = _toy_curves(z)
    f = bcf.BinComboFilter(z=z, curves=curves)

    with pytest.raises(ValueError, match=r"Unknown relation"):
        f.select(spec)
