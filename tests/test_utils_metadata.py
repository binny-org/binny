"""Unit tests for ``binny.utils.metadata``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from binny.utils.metadata import build_tomo_bins_metadata, save_metadata_txt


def test_build_tomo_bins_metadata_includes_optional_notes_and_casts_keys():
    """Tests that build_tomo_bins_metadata includes notes and casts bin keys to int."""
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    parent = np.array([0.0, 1.0, 0.0], dtype=float)
    edges = np.array([0.0, 1.0, 2.0], dtype=float)

    bins = {"1": np.array([0.0, 1.0, 0.0], dtype=float)}

    meta = build_tomo_bins_metadata(
        kind="photoz",
        z=z,
        parent_nz=parent,
        bin_edges=edges,
        bins_returned=bins,
        inputs={"scheme": "x"},
        parent_norm=2.5,
        bins_norms={"1": 7.0},
        frac_per_bin={"1": 0.3},
        density_per_bin={"1": 12.0},
        count_per_bin={"1": 99.0},
        notes={"hello": "world"},
    )

    assert meta["kind"] == "photoz"
    assert meta["grid"]["n"] == 3
    assert meta["bins"]["indices"] == [1]
    assert meta["bins"]["n_bins"] == 1
    assert meta["bins"]["bins_norms"] == {1: 7.0}
    assert meta["bins"]["frac_per_bin"] == {1: 0.3}
    assert meta["bins"]["density_per_bin"] == {1: 12.0}
    assert meta["bins"]["count_per_bin"] == {1: 99.0}
    assert meta["notes"] == {"hello": "world"}


def test_save_metadata_txt_rounds_python_float_numpy_float_and_nested_containers(
    tmp_path: Path,
) -> None:
    """Tests that save_metadata_txt rounds floats in mappings, lists, and tuples."""
    meta = {
        "a": 1.23456,  # python float
        "b": np.float64(2.34567),  # numpy float (np.floating branch)
        "c": {
            "d": 3.33333,
            "e": [4.44444, np.float64(5.55555)],
        },  # Mapping + list
        "f": (6.66666, {"g": 7.77777}),  # tuple + mapping
    }

    out_path = save_metadata_txt(meta, tmp_path / "meta.txt", decimal_places=2)
    assert out_path.exists()

    txt = out_path.read_text(encoding="utf-8")

    # Rounded values should appear in the text dump.
    assert "a: 1.23" in txt
    assert "b: 2.35" in txt
    assert "d: 3.33" in txt
    assert "- 4.44" in txt
    assert "- 5.56" in txt
    assert "- 6.67" in txt
    assert "g: 7.78" in txt


def test_save_metadata_txt_decimal_places_none_keeps_values_verbatim(
    tmp_path: Path,
) -> None:
    """Tests that save_metadata_txt does not round when decimal_places is None."""
    meta = {"x": 1.23456}
    out_path = save_metadata_txt(meta, tmp_path / "meta.txt", decimal_places=None)
    txt = out_path.read_text(encoding="utf-8")

    lines = txt.splitlines()
    assert lines == ["x: 1.23456"]
