"""Integration and validation tests for the functional spec-z builder."""

from __future__ import annotations

import numpy as np
import pytest

from binny.axes.bin_edges import generalized_tomographic_edges
from binny.binning.scheme_registry import register_binning_scheme
from binny.binning.specz import build_specz_bins


def _toy_z_nz(nz_kind: str = "smooth"):
    """Return a synthetic redshift grid and parent distribution."""
    z = np.linspace(0.0, 2.0, 501)

    if nz_kind == "smooth":
        nz = z**2 * np.exp(-z)
    elif nz_kind == "flat":
        nz = np.ones_like(z)
    else:
        raise ValueError("unknown nz_kind")

    return z, nz


def _edges_4bins():
    """Return four equal-width bins spanning redshifts zero to two."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _assert_col_stochastic(
    matrix: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-10,
):
    """Assert that a response matrix is column-stochastic."""
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= -1e-15)
    assert np.allclose(matrix.sum(axis=0), 1.0, rtol=rtol, atol=atol)


def test_build_specz_bins_keys_shapes_and_dtype():
    """Test that bin keys, shapes, and data types are correct."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins = build_specz_bins(z, nz, bins={"edges": edges})

    assert list(bins.keys()) == [0, 1, 2, 3]

    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_normalize_bins_true_integrals_are_one():
    """Test that nonempty bins are normalized to unit integral."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()

    bins = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        normalize_input=True,
        normalize_bins=True,
    )

    for i in range(4):
        integral = np.trapezoid(bins[i], x=z)
        assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-10)


def test_build_specz_bins_no_bin_normalization_respects_completeness():
    """Test that unnormalized bins preserve completeness effects."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()
    completeness = [1.0, 0.8, 0.6, 0.4]

    bins = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        completeness=completeness,
        normalize_input=True,
        normalize_bins=False,
    )

    integrals = np.array([np.trapezoid(bins[i], x=z) for i in range(4)])

    assert np.all(integrals >= 0.0)
    assert integrals.sum() <= 1.0 + 1e-8


def test_build_specz_bins_invalid_edges_outside_range():
    """Test that edges outside the redshift grid are rejected."""
    z, nz = _toy_z_nz()
    edges = np.array([-0.1, 0.5, 1.0])

    with pytest.raises(ValueError, match=r"within"):
        build_specz_bins(z, nz, bins={"edges": edges})


def test_build_specz_bins_invalid_edges_not_increasing():
    """Test that non-increasing edges are rejected."""
    z, nz = _toy_z_nz()
    edges = np.array([0.0, 1.0, 0.9, 2.0])

    with pytest.raises(ValueError, match=r"strictly increasing"):
        build_specz_bins(z, nz, bins={"edges": edges})


def test_build_specz_bins_catastrophic_neighbor_changes_bins():
    """Test that catastrophic neighbor leakage modifies the distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins0 = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )

    bins1 = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.2,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )

    differences = [np.linalg.norm(bins1[i] - bins0[i]) for i in range(4)]

    assert any(difference > 0 for difference in differences)


def test_build_specz_bins_explicit_matrix_overrides_catastrophic_frac():
    """Test that an explicit response matrix controls bin reassignment."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    bins = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        response_matrix=matrix,
        normalize_input=True,
        normalize_bins=False,
    )

    mask_01 = (z >= 0.5) & (z < 1.0)
    mask_00 = (z >= 0.0) & (z < 0.5)

    i0 = np.trapezoid(bins[0][mask_01], x=z[mask_01])
    i1 = np.trapezoid(bins[1][mask_00], x=z[mask_00])

    assert i0 > 0.0
    assert i1 > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_const():
    """Test that constant measurement scatter modifies the distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins_no = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        normalize_bins=False,
    )

    bins_sc = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        specz_scatter=[5e-4, 5e-4, 8e-4, 1e-3],
        specz_scatter_model="const",
        normalize_bins=False,
    )

    distance = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))

    assert distance > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_sigma0_sigma1():
    """Test that redshift-dependent measurement scatter modifies bins."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    bins_no = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        normalize_bins=False,
    )

    bins_sc = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        specz_scatter=None,
        specz_scatter_model="sigma0_plus_sigma1_1pz",
        sigma0=1e-4,
        sigma1=2e-4,
        normalize_bins=False,
    )

    distance = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))

    assert distance > 0.0


def test_build_specz_bins_binning_scheme_equidistant_uses_n_bins():
    """Test that equidistant binning produces the requested bin count."""
    z, nz = _toy_z_nz()

    bins = build_specz_bins(
        z,
        nz,
        bins={"scheme": "equidistant", "n_bins": 4},
    )

    assert list(bins.keys()) == [0, 1, 2, 3]

    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_binning_scheme_equidistant_aliases_work():
    """Test that registered equidistant scheme aliases work."""
    z, nz = _toy_z_nz()

    for scheme in ["eq", "linear"]:
        bins = build_specz_bins(
            z,
            nz,
            bins={"scheme": scheme, "n_bins": 3},
        )

        assert list(bins.keys()) == [0, 1, 2]


def test_build_specz_bins_binning_scheme_equal_number_uses_n_bins():
    """Test that equal-number binning produces the requested bin count."""
    z, nz = _toy_z_nz("smooth")

    bins = build_specz_bins(
        z,
        nz,
        bins={"scheme": "equal_number", "n_bins": 4},
        normalize_input=True,
        normalize_bins=False,
    )

    assert list(bins.keys()) == [0, 1, 2, 3]

    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_binning_scheme_equal_number_aliases_work():
    """Test that registered equal-number scheme aliases work."""
    z, nz = _toy_z_nz("smooth")

    for scheme in ["equipopulated", "en"]:
        bins = build_specz_bins(
            z,
            nz,
            bins={"scheme": scheme, "n_bins": 3},
            normalize_input=True,
            normalize_bins=False,
        )

        assert list(bins.keys()) == [0, 1, 2]


def test_build_specz_bins_mixed_segments_sequence_creates_outputs():
    """Test that a sequence of mixed segments produces the expected bins."""
    z, nz = _toy_z_nz("smooth")

    segments = [
        {
            "scheme": "equidistant",
            "n_bins": 2,
            "z_min": 0.0,
            "z_max": 1.0,
        },
        {
            "scheme": "equidistant",
            "n_bins": 2,
            "z_min": 1.0,
            "z_max": 2.0,
        },
    ]

    bins = build_specz_bins(
        z,
        nz,
        bins={"segments": segments},
    )

    assert list(bins.keys()) == [0, 1, 2, 3]

    for i in range(4):
        assert bins[i].shape == z.shape


def test_build_specz_bins_mixed_segments_dict_with_segments_key_is_accepted():
    """Test that mixed segments can be supplied through a dictionary."""
    z, nz = _toy_z_nz("smooth")

    scheme = {
        "segments": [
            {
                "scheme": "equidistant",
                "n_bins": 2,
                "z_min": 0.0,
                "z_max": 1.0,
            },
            {
                "scheme": "equidistant",
                "n_bins": 1,
                "z_min": 1.0,
                "z_max": 2.0,
            },
        ]
    }

    bins = build_specz_bins(z, nz, bins=scheme)

    assert list(bins.keys()) == [0, 1, 2]

    for i in range(3):
        assert bins[i].shape == z.shape


def test_build_specz_bins_mixed_segments_equal_number_is_accepted():
    """Test that mixed segments support equal-number binning."""
    z, nz = _toy_z_nz("smooth")

    segments = [
        {
            "scheme": "equal_number",
            "n_bins": 4,
            "z_min": 0.0,
            "z_max": 2.0,
        }
    ]

    bins = build_specz_bins(
        z,
        nz,
        bins={"segments": segments},
        normalize_input=True,
        normalize_bins=False,
    )

    assert list(bins.keys()) == [0, 1, 2, 3]


def test_build_specz_bins_rejects_bin_edges_and_binning_scheme_together():
    """Test that explicit edges cannot be combined with a named scheme."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={
                "edges": edges,
                "scheme": "equidistant",
                "n_bins": 4,
            },
        )


def test_build_specz_bins_rejects_bin_edges_and_n_bins_together():
    """Test that explicit edges cannot be combined with a bin count."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={"edges": edges, "n_bins": 4},
        )


def test_build_specz_bins_requires_binning_scheme_when_bin_edges_none():
    """Test that an empty binning specification is rejected."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError):
        build_specz_bins(z, nz, bins={})


def test_build_specz_bins_requires_n_bins_for_string_binning_scheme():
    """Test that a named scheme requires the number of bins."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError, match=r"n_bins"):
        build_specz_bins(
            z,
            nz,
            bins={"scheme": "equidistant"},
        )


def test_build_specz_bins_rejects_unknown_string_binning_scheme():
    """Test that an unknown binning scheme is rejected."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={"scheme": "nope", "n_bins": 4},
        )


def test_build_specz_bins_mixed_mode_rejects_global_n_bins():
    """Test that mixed segments reject a global bin count."""
    z, nz = _toy_z_nz()

    segments = [
        {
            "scheme": "equidistant",
            "n_bins": 2,
            "z_min": 0.0,
            "z_max": 2.0,
        }
    ]

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={"segments": segments, "n_bins": 2},
        )


def test_build_specz_bins_mixed_dict_requires_segments_key():
    """Test that an invalid mixed-binning dictionary is rejected."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={"not_segments": []},
        )


def test_build_specz_bins_mixed_binning_requires_sequence_of_segments():
    """Test that mixed segments must be provided as a sequence."""
    z, nz = _toy_z_nz()

    with pytest.raises(ValueError):
        build_specz_bins(
            z,
            nz,
            bins={"segments": "not a list"},
        )


def test_build_specz_bins_writes_metadata_when_save_path_provided(
    tmp_path,
) -> None:
    """Test that metadata can be saved without returning it."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    outpath = tmp_path / "meta.txt"

    bins = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        include_metadata=False,
        save_metadata_path=str(outpath),
    )

    assert isinstance(bins, dict)
    assert outpath.exists()

    text = outpath.read_text(encoding="utf-8")
    assert "kind" in text or "specz" in text


def test_build_specz_bins_warns_when_zero_catastrophic_frac_and_leakage_params_passed() -> None:
    """Test that unused catastrophic leakage parameters produce a warning."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()

    with pytest.warns(
        RuntimeWarning,
        match=(r"catastrophic_frac is zero for all bins, " r"so catastrophic leakage parameters"),
    ):
        bins = build_specz_bins(
            z,
            nz,
            bins={"edges": edges},
            catastrophic_frac=0.0,
            leakage_model="uniform",
            leakage_sigma=3.0,
            normalize_input=True,
            normalize_bins=False,
        )

    assert list(bins.keys()) == [0, 1, 2, 3]

    for i in range(4):
        assert bins[i].shape == z.shape


def test_build_specz_bins_generalized() -> None:
    """Test generalized binning against explicitly supplied edges."""
    z, nz = _toy_z_nz("smooth")

    dchi_dz = 1.0 / np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)

    chi = np.zeros_like(z)
    chi[1:] = np.cumsum(0.5 * (dchi_dz[1:] + dchi_dz[:-1]) * np.diff(z))

    alpha = 0.5
    beta = 0.5
    n_bins = 5

    edges = generalized_tomographic_edges(
        z,
        nz,
        n_bins,
        alpha=alpha,
        beta=beta,
        comoving_distance=chi,
    )

    bins_generalized = build_specz_bins(
        z,
        nz,
        bins={
            "scheme": "generalized",
            "n_bins": n_bins,
            "params": {
                "alpha": alpha,
                "beta": beta,
                "comoving_distance": chi,
            },
        },
    )

    bins_explicit = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
    )

    assert list(bins_generalized) == list(bins_explicit) == list(range(n_bins))

    for i in range(n_bins):
        np.testing.assert_allclose(
            bins_generalized[i],
            bins_explicit[i],
            rtol=1e-12,
            atol=1e-14,
        )


def test_build_specz_bins_custom_scheme() -> None:
    """Test registered custom schemes against explicit bin edges."""
    z, nz = _toy_z_nz("smooth")
    n_bins = 4

    @register_binning_scheme("test_specz_custom_power")
    def custom_scheme(z, nz, n_bins, *, power=2.0):
        """Return custom power-law-spaced bin edges."""
        t = np.linspace(0.0, 1.0, n_bins + 1)
        return z[0] + (z[-1] - z[0]) * t**power

    edges = custom_scheme(z, nz, n_bins, power=2.0)

    bins_custom = build_specz_bins(
        z,
        nz,
        bins={
            "scheme": "test_specz_custom_power",
            "n_bins": n_bins,
            "params": {"power": 2.0},
        },
    )

    bins_explicit = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
    )

    assert list(bins_custom) == list(bins_explicit) == list(range(n_bins))

    for i in range(n_bins):
        np.testing.assert_allclose(
            bins_custom[i],
            bins_explicit[i],
            rtol=1e-12,
            atol=1e-14,
        )
