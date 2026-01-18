"""Unit tests for ``binny.utils.io``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from binny.utils.io import (
    load_binning_recipe,
    load_nz,
    load_yaml,
)


def _write_txt(path: Path, arr: np.ndarray, *, delimiter: str | None = None) -> None:
    """Writes a 2D array to a text file."""
    if delimiter is None:
        np.savetxt(path, arr)  # default whitespace delimiter
    else:
        np.savetxt(path, arr, delimiter=delimiter)


def test_load_nz_npy_two_column_array_sorts_by_z(tmp_path: Path) -> None:
    """Tests that load_nz loads .npy (N,2) and sorts by ascending z."""
    p = tmp_path / "nz.npy"
    arr = np.array(
        [
            [2.0, 20.0],
            [0.0, 0.0],
            [1.0, 10.0],
        ]
    )
    np.save(p, arr)

    z, nz = load_nz(p)
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_npy_two_column_array_respects_column_indices(
    tmp_path: Path,
) -> None:
    """Tests that load_nz respects x_col and nz_col for 2D arrays."""
    p = tmp_path / "nz.npy"
    arr = np.array(
        [
            [999.0, 2.0, 20.0],
            [999.0, 0.0, 0.0],
            [999.0, 1.0, 10.0],
        ]
    )
    np.save(p, arr)

    z, nz = load_nz(p, x_col=1, nz_col=2)
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_npy_structured_fields_z_nz(tmp_path: Path) -> None:
    """Tests that load_nz loads .npy structured array with 'z' and 'nz' fields."""
    p = tmp_path / "nz_struct.npy"
    arr = np.zeros(3, dtype=[("z", "f8"), ("nz", "f8")])
    arr["z"] = np.array([2.0, 0.0, 1.0])
    arr["nz"] = np.array([20.0, 0.0, 10.0])
    np.save(p, arr)

    z, nz = load_nz(p)
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_npy_structured_missing_fields_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises if structured .npy lacks required fields."""
    p = tmp_path / "bad_struct.npy"
    arr = np.zeros(3, dtype=[("z", "f8"), ("nope", "f8")])
    np.save(p, arr)

    with pytest.raises(ValueError, match=r"(does not contain.*z.*nz|no field of name nz)"):
        load_nz(p)


def test_load_nz_npy_unsupported_structure_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises on unsupported .npy structure."""
    p = tmp_path / "bad.npy"
    np.save(p, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match=r"Unsupported \.npy structure"):
        load_nz(p)


def test_load_nz_npz_with_key_loads_array(tmp_path: Path) -> None:
    """Tests that load_nz loads .npz using a provided key."""
    p = tmp_path / "nz.npz"
    arr = np.array([[0.0, 0.0], [2.0, 20.0], [1.0, 10.0]])
    np.savez(p, mykey=arr)

    z, nz = load_nz(p, key="mykey")
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_npz_missing_key_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises if key is missing in .npz."""
    p = tmp_path / "nz.npz"
    np.savez(p, other=np.array([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match=r"Key .* not found"):
        load_nz(p, key="mykey")


def test_load_nz_npz_key_array_wrong_shape_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises if keyed array in .npz is not (N,2+)."""
    p = tmp_path / "nz.npz"
    np.savez(p, mykey=np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match=r"must have shape \(N, 2\)"):
        load_nz(p, key="mykey")


def test_load_nz_npz_without_key_picks_first_suitable(tmp_path: Path) -> None:
    """Tests that load_nz picks the first (N,2+) array if key is None."""
    p = tmp_path / "nz.npz"
    bad = np.array([1.0, 2.0, 3.0])  # not suitable
    good = np.array([[2.0, 20.0], [0.0, 0.0], [1.0, 10.0]])
    np.savez(p, a=bad, b=good)

    z, nz = load_nz(p)
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_npz_without_key_no_suitable_array_raises(
    tmp_path: Path,
) -> None:
    """Tests that load_nz raises if .npz has no (N,2+) array and no key is given."""
    p = tmp_path / "nz.npz"
    np.savez(p, a=np.array([1.0, 2.0]), b=np.array([[1.0, 2.0, 3.0]]))  # (1,3) ok actually
    # Ensure neither is suitable: use (1,) and (1,1)
    p2 = tmp_path / "nz2.npz"
    np.savez(p2, a=np.array([1.0, 2.0]), b=np.array([[1.0]]))

    with pytest.raises(ValueError, match=r"No suitable \(N,2\) array found"):
        load_nz(p2)


def test_load_nz_text_two_columns_loads_and_sorts(tmp_path: Path) -> None:
    """Tests that load_nz loads text files and sorts by z."""
    p = tmp_path / "nz.txt"
    arr = np.array([[2.0, 20.0], [0.0, 0.0], [1.0, 10.0]])
    _write_txt(p, arr)

    z, nz = load_nz(p)
    assert np.allclose(z, [0.0, 1.0, 2.0])
    assert np.allclose(nz, [0.0, 10.0, 20.0])


def test_load_nz_text_one_column_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises if a text file has only one column."""
    p = tmp_path / "nz.txt"
    np.savetxt(p, np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match=r"must have at least two columns"):
        load_nz(p)


def test_load_nz_text_out_of_bounds_columns_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises if requested columns are out of bounds for text."""
    p = tmp_path / "nz.txt"
    arr = np.array([[0.0, 0.0], [1.0, 1.0]])
    _write_txt(p, arr)

    with pytest.raises(ValueError, match=r"out of bounds"):
        load_nz(p, x_col=0, nz_col=2)


def test_load_nz_rejects_nonfinite_z(tmp_path: Path) -> None:
    """Tests that load_nz raises if loaded z contains non-finite values."""
    p = tmp_path / "nz.npy"
    arr = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])
    np.save(p, arr)

    with pytest.raises(ValueError, match=r"Loaded z contains non-finite"):
        load_nz(p)


def test_load_nz_rejects_nonfinite_nz(tmp_path: Path) -> None:
    """Tests that load_nz raises if loaded nz contains non-finite values."""
    p = tmp_path / "nz.npy"
    arr = np.array([[0.0, 0.0], [1.0, np.inf], [2.0, 2.0]])
    np.save(p, arr)

    with pytest.raises(ValueError, match=r"Loaded nz contains non-finite"):
        load_nz(p)


def test_load_nz_unsupported_extension_raises(tmp_path: Path) -> None:
    """Tests that load_nz raises for unsupported file extensions."""
    p = tmp_path / "nz.fits"
    p.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unsupported file extension"):
        load_nz(p)


def test_load_binning_recipe_valid_yaml_returns_normalized_segments(
    tmp_path: Path,
) -> None:
    """Tests that load_binning_recipe returns normalized segment dicts."""
    p = tmp_path / "recipe.yml"
    data = {
        "name": "example",
        "n_bins": 5,
        "segments": [
            {
                "method": "eq",
                "n_bins": 3,
                "params": {"x_min": 0.0, "x_max": 1.0},
            },
            {"method": "equal_number", "n_bins": 2},
        ],
    }
    p.write_text(yaml.safe_dump(data), encoding="utf-8")

    segs = load_binning_recipe(str(p))
    assert isinstance(segs, list)
    assert segs[0]["method"] == "eq"
    assert segs[0]["n_bins"] == 3
    assert segs[0]["params"] == {"x_min": 0.0, "x_max": 1.0}
    assert segs[1]["method"] == "equal_number"
    assert segs[1]["n_bins"] == 2
    assert segs[1]["params"] == {}


def test_load_binning_recipe_top_level_not_mapping_raises(
    tmp_path: Path,
) -> None:
    """Tests that load_binning_recipe raises if YAML top-level is not a mapping."""
    p = tmp_path / "recipe.yml"
    p.write_text(yaml.safe_dump(["nope"]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Top-level YAML content must be a mapping"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_missing_segments_raises(tmp_path: Path) -> None:
    """Tests that load_binning_recipe raises if 'segments' is missing."""
    p = tmp_path / "recipe.yml"
    p.write_text(yaml.safe_dump({"n_bins": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"non-empty 'segments'"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_empty_segments_raises(tmp_path: Path) -> None:
    """Tests that load_binning_recipe raises if 'segments' is empty."""
    p = tmp_path / "recipe.yml"
    p.write_text(yaml.safe_dump({"segments": []}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"non-empty 'segments'"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_segment_not_mapping_raises(tmp_path: Path) -> None:
    """Tests that load_binning_recipe raises if any segment is not a mapping."""
    p = tmp_path / "recipe.yml"
    p.write_text(yaml.safe_dump({"segments": ["nope"]}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Segment 0 must be a mapping"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_segment_missing_required_keys_raises(
    tmp_path: Path,
) -> None:
    """Tests that load_binning_recipe raises if a segment lacks method or n_bins."""
    p = tmp_path / "recipe.yml"
    p.write_text(yaml.safe_dump({"segments": [{"method": "eq"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"must contain at least 'method' and 'n_bins'"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_params_not_mapping_raises(tmp_path: Path) -> None:
    """Tests that load_binning_recipe raises if segment params is not a mapping."""
    p = tmp_path / "recipe.yml"
    p.write_text(
        yaml.safe_dump({"segments": [{"method": "eq", "n_bins": 1, "params": ["x"]}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"'params' must be a mapping"):
        load_binning_recipe(str(p))


def test_load_binning_recipe_validate_mixed_segments_errors_propagate(
    tmp_path: Path,
) -> None:
    """Tests that load_binning_recipe propagates validation failures."""
    p = tmp_path / "recipe.yml"
    # total_n_bins=3 but segments sum to 4 -> should fail validation
    p.write_text(
        yaml.safe_dump(
            {
                "n_bins": 3,
                "segments": [
                    {"method": "eq", "n_bins": 2},
                    {"method": "eq", "n_bins": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_binning_recipe(str(p))


def test_load_yaml_from_disk_mapping_returns_dict(tmp_path: Path) -> None:
    """Tests that load_yaml loads a YAML mapping from disk."""
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump({"a": 1, "b": {"c": 2}}), encoding="utf-8")

    out = load_yaml(p)
    assert isinstance(out, dict)
    assert out["a"] == 1
    assert out["b"]["c"] == 2


def test_load_yaml_from_disk_non_mapping_raises(tmp_path: Path) -> None:
    """Tests that load_yaml raises when disk YAML root is not a mapping."""
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(["nope"]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Top-level YAML content must be a mapping"):
        load_yaml(p)


def test_load_yaml_from_package_mapping(monkeypatch, tmp_path: Path) -> None:
    """Tests that load_yaml loads a YAML mapping from a package resource."""
    # Create a fake package resource dir with a YAML file
    pkgdir = tmp_path / "fakepkg"
    pkgdir.mkdir()
    (pkgdir / "__init__.py").write_text("", encoding="utf-8")
    (pkgdir / "x.yml").write_text(yaml.safe_dump({"k": "v"}), encoding="utf-8")

    # Patch importlib.resources.files to point at our fake package dir
    import importlib.resources as resources

    def fake_files(package: str):
        assert package == "fakepkg"
        return pkgdir

    monkeypatch.setattr(resources, "files", fake_files)

    out = load_yaml("x.yml", package="fakepkg")
    assert out == {"k": "v"}


def test_load_yaml_from_package_non_mapping_raises(monkeypatch, tmp_path: Path) -> None:
    """Tests that load_yaml raises when packaged YAML root is not a mapping."""
    pkgdir = tmp_path / "fakepkg"
    pkgdir.mkdir()
    (pkgdir / "__init__.py").write_text("", encoding="utf-8")
    (pkgdir / "x.yml").write_text(yaml.safe_dump(["nope"]), encoding="utf-8")

    import importlib.resources as resources

    def fake_files(package: str):
        assert package == "fakepkg"
        return pkgdir

    monkeypatch.setattr(resources, "files", fake_files)

    with pytest.raises(ValueError, match=r"Top-level YAML content must be a mapping"):
        load_yaml("x.yml", package="fakepkg")
