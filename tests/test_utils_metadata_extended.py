"""Unit tests for ``binny.utils.metadata`` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from binny.utils.metadata import build_tomo_bins_metadata, save_metadata_txt


def _toy_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Returns toy z, parent_nz, edges, and bins_returned."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)
    parent = np.array([1.0, 2.0, 1.0], dtype=float)
    edges = np.array([0.0, 0.6, 1.0], dtype=float)
    bins = {
        0: np.array([0.0, 1.0, 0.0], dtype=float),
        1: np.array([0.0, 0.5, 0.5], dtype=float),
    }
    return z, parent, edges, bins


def test_build_tomo_bins_metadata_minimal_fields_and_types() -> None:
    """Tests that build_tomo_bins_metadata returns expected minimal structure."""
    z, parent, edges, bins = _toy_inputs()
    inputs = {"scheme": "equidistant", "n_bins": 2}

    meta = build_tomo_bins_metadata(
        kind="photoz",
        z=z,
        parent_nz=parent,
        bin_edges=edges,
        bins_returned=bins,
        inputs=inputs,
    )

    assert meta["kind"] == "photoz"

    grid = meta["grid"]
    assert grid["n"] == 3
    assert grid["z_min"] == 0.0
    assert grid["z_max"] == 1.0
    assert grid["z"] == z.tolist()

    parent_nz = meta["parent_nz"]
    assert parent_nz["values"] == parent.tolist()
    assert parent_nz["norm"] is None

    b = meta["bins"]
    assert b["indices"] == [0, 1]
    assert b["n_bins"] == 2
    assert b["bin_edges"] == edges.tolist()

    assert isinstance(b["bins_returned"], dict)
    assert b["bins_returned"][0] == bins[0].tolist()
    assert b["bins_returned"][1] == bins[1].tolist()

    assert b["bins_norms"] is None
    assert b["frac_per_bin"] is None
    assert b["density_per_bin"] is None
    assert b["count_per_bin"] is None

    assert meta["inputs"] == inputs


def test_build_tomo_bins_metadata_optional_fields_are_cast_and_present() -> None:
    """Tests that build_tomo_bins_metadata includes optional population fields."""
    z, parent, edges, bins = _toy_inputs()
    inputs = {"scheme": "equidistant", "n_bins": 2}

    meta = build_tomo_bins_metadata(
        kind="specz",
        z=z,
        parent_nz=parent,
        bin_edges=edges,
        bins_returned=bins,
        inputs=inputs,
        parent_norm=3,
        bins_norms={0: 1, 1: 2.0},
        frac_per_bin={0: 0.25, 1: 0.75},
        density_per_bin={0: 10, 1: 20},
        count_per_bin={0: 100, 1: 200},
        notes={"hello": "world", "x": 1},
    )

    assert meta["kind"] == "specz"
    assert meta["parent_nz"]["norm"] == 3.0

    b = meta["bins"]
    assert b["bins_norms"] == {0: 1.0, 1: 2.0}
    assert b["frac_per_bin"] == {0: 0.25, 1: 0.75}
    assert b["density_per_bin"] == {0: 10.0, 1: 20.0}
    assert b["count_per_bin"] == {0: 100.0, 1: 200.0}

    assert meta["notes"] == {"hello": "world", "x": 1}


def test_build_tomo_bins_metadata_handles_empty_grid() -> None:
    """Tests that build_tomo_bins_metadata handles empty z gracefully."""
    z = np.array([], dtype=float)
    parent = np.array([], dtype=float)
    edges = np.array([0.0, 1.0], dtype=float)
    bins = {0: np.array([], dtype=float)}
    inputs = {"scheme": "equidistant", "n_bins": 1}

    meta = build_tomo_bins_metadata(
        kind="photoz",
        z=z,
        parent_nz=parent,
        bin_edges=edges,
        bins_returned=bins,
        inputs=inputs,
    )

    assert meta["grid"]["n"] == 0
    assert meta["grid"]["z_min"] is None
    assert meta["grid"]["z_max"] is None
    assert meta["grid"]["z"] == []


def test_save_metadata_txt_writes_deterministic_sorted_keys(
    tmp_path: Path,
) -> None:
    """Tests that save_metadata_txt writes a deterministic, readable text dump."""
    # Purposefully unsorted keys to test deterministic sorting.
    meta = {"b": 2, "a": {"y": 1, "x": 0}, "c": [3, {"k": 9, "j": 8}]}

    out = save_metadata_txt(meta, tmp_path / "meta.txt")
    assert out.exists()

    text = out.read_text(encoding="utf-8")
    # Top-level keys sorted: a, b, c
    lines = text.splitlines()
    assert lines[0] == "a:"
    assert lines[1].startswith("  x:")
    assert lines[2].startswith("  y:")
    assert lines[3] == "b: 2"
    assert lines[4] == "c:"
    # Ensure newline at end (save_metadata_txt adds one).
    assert text.endswith("\n")


def test_save_metadata_txt_accepts_str_path(tmp_path: Path) -> None:
    """Tests that save_metadata_txt accepts a string path."""
    meta = {"a": 1}
    p = tmp_path / "meta2.txt"

    out = save_metadata_txt(meta, str(p))
    assert isinstance(out, Path)
    assert out == p
    assert p.read_text(encoding="utf-8") == "a: 1\n"


def test_build_tomo_bins_metadata_does_not_mutate_inputs_mapping() -> None:
    """Tests that build_tomo_bins_metadata does not mutate the inputs mapping."""
    z, parent, edges, bins = _toy_inputs()
    inputs = {"scheme": "equidistant", "n_bins": 2}
    before = dict(inputs)

    _ = build_tomo_bins_metadata(
        kind="photoz",
        z=z,
        parent_nz=parent,
        bin_edges=edges,
        bins_returned=bins,
        inputs=inputs,
    )

    assert inputs == before
