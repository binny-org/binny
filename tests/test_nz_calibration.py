"""Unit tests for `binny.nz.calibration`."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz.calibration import (
    _smail_logpdf,
    calibrate_depth_smail_from_mock,
    eval_ngal_of_maglim,
    eval_z0_of_maglim,
    fit_ngal_of_maglim_from_mock,
    fit_smail_params_from_mock,
    fit_z0_of_maglim_from_mock,
)


def _make_smail_samples(
    n: int = 4000,
    *,
    alpha: float = 2.0,
    beta: float = 1.5,
    z0: float = 0.35,
    seed: int = 123,
) -> np.ndarray:
    """Tests use a reproducible Smail-distributed sample."""
    rng = np.random.default_rng(seed)
    t = rng.gamma(shape=(alpha + 1.0) / beta, scale=1.0, size=n)
    return z0 * t ** (1.0 / beta)


def _make_mock_catalog(
    n: int = 5000,
    *,
    alpha: float = 2.0,
    beta: float = 1.5,
    z0_base: float = 0.20,
    mag0: float = 22.0,
    mag_slope: float = 3.0,
    mag_noise: float = 0.35,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Tests use a reproducible mock catalog with correlated z and magnitude."""
    rng = np.random.default_rng(seed)
    z = _make_smail_samples(n=n, alpha=alpha, beta=beta, z0=z0_base, seed=seed)
    mag = mag0 + mag_slope * z + rng.normal(0.0, mag_noise, size=n)
    return z, mag


def test_smail_logpdf_returns_same_shape_as_input():
    """Tests that _smail_logpdf returns an array with the input shape."""
    z = np.array([0.0, 0.1, 0.5, 1.0])
    out = _smail_logpdf(z, z0=0.3, alpha=2.0, beta=1.5)

    assert isinstance(out, np.ndarray)
    assert out.shape == z.shape


def test_smail_logpdf_returns_minus_inf_for_invalid_parameters():
    """Tests that _smail_logpdf returns minus infinity for invalid parameters."""
    z = np.array([0.0, 0.2, 0.4])

    out1 = _smail_logpdf(z, z0=-1.0, alpha=2.0, beta=1.5)
    out2 = _smail_logpdf(z, z0=0.3, alpha=-1.1, beta=1.5)
    out3 = _smail_logpdf(z, z0=0.3, alpha=2.0, beta=0.0)

    assert np.all(out1 == -np.inf)
    assert np.all(out2 == -np.inf)
    assert np.all(out3 == -np.inf)


def test_smail_logpdf_returns_minus_inf_for_negative_redshift_only():
    """Tests that _smail_logpdf returns minus infinity only for negative redshifts."""
    z = np.array([-0.1, 0.0, 0.2, 1.0])
    out = _smail_logpdf(z, z0=0.3, alpha=2.0, beta=1.5)

    assert out[0] == -np.inf
    assert np.all(np.isfinite(out[1:]))


def test_smail_logpdf_is_normalized_after_exponentiation():
    """Tests that exp(_smail_logpdf) integrates to about one."""
    z = np.linspace(0.0, 5.0, 20000)
    pdf = np.exp(_smail_logpdf(z, z0=0.4, alpha=2.0, beta=1.5))
    integral = np.trapezoid(pdf, z)

    assert integral == pytest.approx(1.0, rel=5e-3, abs=5e-3)


def test_fit_smail_params_from_mock_returns_failure_for_too_few_samples():
    """Tests that fit_smail_params_from_mock returns failure for too few samples."""
    z = np.array([0.1, 0.2, 0.3])

    out = fit_smail_params_from_mock(z, min_n=10)

    assert out["ok"] is False
    assert out["reason"] == "too_few_samples"
    assert out["params"] is None
    assert out["n"] == 3


def test_fit_smail_params_from_mock_filters_nonfinite_and_negative_values():
    """Tests that fit_smail_params_from_mock filters invalid redshift values."""
    z_good = _make_smail_samples(n=600, seed=1)
    z = np.concatenate([z_good, [np.nan, np.inf, -np.inf, -0.5]])

    out = fit_smail_params_from_mock(z, min_n=200)

    assert out["ok"] is True
    assert out["n"] == 600
    assert set(out["params"]) == {"alpha", "beta", "z0"}


def test_fit_smail_params_from_mock_applies_z_max_cut():
    """Tests that fit_smail_params_from_mock applies the z_max cut."""
    z = np.concatenate([_make_smail_samples(n=500, seed=2), np.array([10.0, 12.0, 15.0])])

    out_no_cut = fit_smail_params_from_mock(z, min_n=200)
    out_cut = fit_smail_params_from_mock(z, z_max=3.0, min_n=200)

    assert out_no_cut["ok"] is True
    assert out_cut["ok"] is True
    assert out_no_cut["n"] == 503
    assert out_cut["n"] < out_no_cut["n"]


def test_fit_smail_params_from_mock_recovers_parameters_reasonably_well():
    """Tests that fit_smail_params_from_mock approximately recovers Smail parameters."""
    alpha_true = 2.0
    beta_true = 1.5
    z0_true = 0.35
    z = _make_smail_samples(
        n=12000,
        alpha=alpha_true,
        beta=beta_true,
        z0=z0_true,
        seed=10,
    )

    out = fit_smail_params_from_mock(z, min_n=200)

    assert out["ok"] is True
    assert out["method"] == "mle_l_bfgs_b"
    assert out["n"] == 12000
    assert out["params"]["alpha"] == pytest.approx(alpha_true, rel=0.15)
    assert out["params"]["beta"] == pytest.approx(beta_true, rel=0.15)
    assert out["params"]["z0"] == pytest.approx(z0_true, rel=0.12)


def test_fit_smail_params_from_mock_uses_custom_initial_guess():
    """Tests that fit_smail_params_from_mock accepts a custom initial guess."""
    z = _make_smail_samples(n=3000, alpha=2.3, beta=1.2, z0=0.4, seed=4)

    out = fit_smail_params_from_mock(z, x0=(1.0, 1.0, 0.2), min_n=200)

    assert out["ok"] is True
    assert out["params"] is not None


def test_fit_z0_of_maglim_from_mock_returns_failure_when_too_few_valid_points():
    """Tests that fit_z0_of_maglim_from_mock fails when too few cuts are usable."""
    z, mag = _make_mock_catalog(n=300, seed=11)
    maglims = np.array([10.0, 10.5, 11.0])

    out = fit_z0_of_maglim_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        alpha=2.0,
        beta=1.5,
        min_n_per_cut=100,
    )

    assert out["ok"] is False
    assert out["reason"] == "too_few_valid_points"
    assert out["points"] is None
    assert out["fit"] is None


def test_fit_z0_of_maglim_from_mock_returns_linear_fit_and_points():
    """Tests that fit_z0_of_maglim_from_mock returns linear fit metadata and points."""
    z, mag = _make_mock_catalog(n=8000, seed=12)
    maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

    out = fit_z0_of_maglim_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        alpha=2.0,
        beta=1.5,
        z0_law="linear",
        min_n_per_cut=100,
    )

    assert out["ok"] is True
    assert out["fit"]["law"] == "linear"
    assert set(out["fit"]) == {"law", "a", "b"}
    assert np.allclose(out["points"]["maglim"], maglims)
    assert out["points"]["z0"].shape == maglims.shape
    assert out["points"]["n_selected"].shape == maglims.shape
    assert np.all(out["points"]["n_selected"] >= 0)
    assert np.all(np.diff(out["points"]["n_selected"]) >= 0)


def test_fit_z0_of_maglim_from_mock_returns_poly2_fit():
    """Tests that fit_z0_of_maglim_from_mock returns a quadratic fit when requested."""
    z, mag = _make_mock_catalog(n=8000, seed=13)
    maglims = np.array([22.3, 22.8, 23.3, 23.8, 24.3, 24.8])

    out = fit_z0_of_maglim_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        alpha=2.0,
        beta=1.5,
        z0_law="poly2",
        min_n_per_cut=100,
    )

    assert out["ok"] is True
    assert out["fit"]["law"] == "poly2"
    assert set(out["fit"]) == {"law", "c2", "c1", "c0"}


def test_fit_z0_of_maglim_from_mock_rejects_unknown_law():
    """Tests that fit_z0_of_maglim_from_mock rejects an unknown z0 law."""
    z, mag = _make_mock_catalog(n=4000, seed=14)
    maglims = np.array([22.5, 23.0, 23.5, 24.0])

    with pytest.raises(ValueError, match=r"z0_law must be 'linear' or 'poly2'"):
        fit_z0_of_maglim_from_mock(
            z_true=z,
            mag=mag,
            maglims=maglims,
            alpha=2.0,
            beta=1.5,
            z0_law="banana",
            min_n_per_cut=50,
        )


def test_fit_z0_of_maglim_from_mock_applies_z_max_cut():
    """Tests that fit_z0_of_maglim_from_mock respects the z_max cut."""
    z, mag = _make_mock_catalog(n=7000, seed=15)
    maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

    out_no_cut = fit_z0_of_maglim_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        alpha=2.0,
        beta=1.5,
        z0_law="linear",
        min_n_per_cut=100,
    )
    out_cut = fit_z0_of_maglim_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        alpha=2.0,
        beta=1.5,
        z0_law="linear",
        z_max=0.8,
        min_n_per_cut=100,
    )

    assert out_no_cut["ok"] is True
    assert out_cut["ok"] is True
    assert not np.allclose(out_no_cut["points"]["z0"], out_cut["points"]["z0"], equal_nan=True)


def test_eval_z0_of_maglim_evaluates_linear_fit():
    """Tests that eval_z0_of_maglim evaluates a linear fit correctly."""
    maglim = np.array([24.0, 25.0, 26.0])
    fit = {"law": "linear", "a": 0.1, "b": -2.0}

    out = eval_z0_of_maglim(maglim, fit)

    expected = np.array([0.4, 0.5, 0.6])
    assert np.allclose(out, expected)


def test_eval_z0_of_maglim_evaluates_poly2_fit():
    """Tests that eval_z0_of_maglim evaluates a quadratic fit correctly."""
    maglim = np.array([1.0, 2.0, 3.0])
    fit = {"law": "poly2", "c2": 2.0, "c1": -1.0, "c0": 0.5}

    out = eval_z0_of_maglim(maglim, fit)

    expected = np.array([1.5, 6.5, 15.5])
    assert np.allclose(out, expected)


def test_eval_z0_of_maglim_rejects_unknown_law():
    """Tests that eval_z0_of_maglim rejects an unknown law."""
    with pytest.raises(ValueError, match=r"Unknown law"):
        eval_z0_of_maglim(np.array([24.0]), {"law": "weird"})


def test_fit_ngal_of_maglim_from_mock_rejects_nonpositive_area():
    """Tests that fit_ngal_of_maglim_from_mock rejects nonpositive area."""
    mag = np.array([20.0, 21.0, 22.0])
    maglims = np.array([21.0, 22.0])

    with pytest.raises(ValueError, match=r"area_deg2 must be > 0"):
        fit_ngal_of_maglim_from_mock(mag, maglims=maglims, area_deg2=0.0)


def test_fit_ngal_of_maglim_from_mock_ignores_nonfinite_magnitudes():
    """Tests that fit_ngal_of_maglim_from_mock ignores nonfinite magnitudes."""
    mag = np.array([20.0, 21.0, 22.0, np.nan, np.inf, -np.inf])
    maglims = np.array([21.0, 22.0, 23.0])

    out = fit_ngal_of_maglim_from_mock(
        mag,
        maglims=maglims,
        area_deg2=1.0,
        ngal_law="linear",
    )

    assert out["ok"] is True
    assert np.array_equal(out["points"]["n_selected"], np.array([2, 3, 3]))


def test_fit_ngal_of_maglim_from_mock_returns_failure_for_too_few_positive_points():
    """Tests that fit_ngal_of_maglim_from_mock fails when too few ngal points are positive."""
    mag = np.array([30.0, 31.0, 32.0])
    maglims = np.array([20.0, 21.0, 22.0])

    out = fit_ngal_of_maglim_from_mock(
        mag,
        maglims=maglims,
        area_deg2=1.0,
        ngal_law="linear",
    )

    assert out["ok"] is False
    assert out["reason"] == "too_few_valid_points"
    assert out["points"] is None
    assert out["fit"] is None


def test_fit_ngal_of_maglim_from_mock_returns_linear_fit():
    """Tests that fit_ngal_of_maglim_from_mock returns a linear fit when requested."""
    _, mag = _make_mock_catalog(n=5000, seed=16)
    maglims = np.array([22.0, 22.5, 23.0, 23.5, 24.0])

    out = fit_ngal_of_maglim_from_mock(
        mag,
        maglims=maglims,
        area_deg2=2.0,
        ngal_law="linear",
    )

    assert out["ok"] is True
    assert out["fit"]["law"] == "linear"
    assert set(out["fit"]) == {"law", "p", "q"}
    assert np.allclose(out["points"]["maglim"], maglims)
    assert np.all(np.diff(out["points"]["n_selected"]) >= 0)
    assert np.all(np.diff(out["points"]["ngal_arcmin2"]) >= 0)


def test_fit_ngal_of_maglim_from_mock_returns_loglinear_fit():
    """Tests that fit_ngal_of_maglim_from_mock returns a loglinear fit by default."""
    _, mag = _make_mock_catalog(n=5000, seed=17)
    maglims = np.array([22.0, 22.5, 23.0, 23.5, 24.0])

    out = fit_ngal_of_maglim_from_mock(
        mag,
        maglims=maglims,
        area_deg2=2.0,
    )

    assert out["ok"] is True
    assert out["fit"]["law"] == "loglinear"
    assert set(out["fit"]) == {"law", "s", "t"}


def test_fit_ngal_of_maglim_from_mock_rejects_unknown_law():
    """Tests that fit_ngal_of_maglim_from_mock rejects an unknown ngal law."""
    _, mag = _make_mock_catalog(n=1000, seed=18)
    maglims = np.array([22.0, 23.0, 24.0])

    with pytest.raises(ValueError, match=r"ngal_law must be 'linear' or 'loglinear'"):
        fit_ngal_of_maglim_from_mock(
            mag,
            maglims=maglims,
            area_deg2=1.0,
            ngal_law="banana",
        )


def test_eval_ngal_of_maglim_evaluates_linear_fit():
    """Tests that eval_ngal_of_maglim evaluates a linear fit correctly."""
    maglim = np.array([24.0, 25.0, 26.0])
    fit = {"law": "linear", "p": 2.0, "q": -10.0}

    out = eval_ngal_of_maglim(maglim, fit)

    expected = np.array([38.0, 40.0, 42.0])
    assert np.allclose(out, expected)


def test_eval_ngal_of_maglim_evaluates_loglinear_fit():
    """Tests that eval_ngal_of_maglim evaluates a loglinear fit correctly."""
    maglim = np.array([20.0, 21.0, 22.0])
    fit = {"law": "loglinear", "s": 0.5, "t": -1.0}

    out = eval_ngal_of_maglim(maglim, fit)

    expected = 10.0 ** (0.5 * maglim - 1.0)
    assert np.allclose(out, expected)


def test_eval_ngal_of_maglim_rejects_unknown_law():
    """Tests that eval_ngal_of_maglim rejects an unknown law."""
    with pytest.raises(ValueError, match=r"Unknown law"):
        eval_ngal_of_maglim(np.array([24.0]), {"law": "weird"})


def test_calibrate_depth_smail_from_mock_runs_end_to_end_with_deep_cut():
    """Tests that calibrate_depth_smail_from_mock works with deep-cut inference."""
    z, mag = _make_mock_catalog(n=12000, seed=19)
    maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

    out = calibrate_depth_smail_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        area_deg2=5.0,
        infer_alpha_beta_from="deep_cut",
        alpha_beta_maglim=24.5,
        z_max=3.0,
    )

    assert out["ok"] is True
    assert out["alpha_beta_fit"]["ok"] is True
    assert out["z0_of_maglim"]["ok"] is True
    assert out["ngal_of_maglim"]["ok"] is True


def test_calibrate_depth_smail_from_mock_runs_end_to_end_with_all_selected():
    """Tests that calibrate_depth_smail_from_mock works with all-selected inference."""
    z, mag = _make_mock_catalog(n=12000, seed=20)
    maglims = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

    out = calibrate_depth_smail_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        area_deg2=5.0,
        infer_alpha_beta_from="all_selected_at_maglim",
        z_max=3.0,
    )

    assert out["ok"] is True
    assert out["alpha_beta_fit"]["ok"] is True
    assert out["z0_of_maglim"]["ok"] is True
    assert out["ngal_of_maglim"]["ok"] is True


def test_calibrate_depth_smail_from_mock_defaults_alpha_beta_maglim_to_max_maglim():
    """Tests that calibrate_depth_smail_from_mock defaults alpha-beta maglim to the max maglim."""
    z, mag = _make_mock_catalog(n=9000, seed=21)
    maglims = np.array([22.0, 23.0, 24.0])

    out = calibrate_depth_smail_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        area_deg2=4.0,
        infer_alpha_beta_from="deep_cut",
        alpha_beta_maglim=None,
    )

    assert out["alpha_beta_fit"]["ok"] is True
    assert out["z0_of_maglim"]["ok"] is True
    assert out["ngal_of_maglim"]["ok"] is True


def test_calibrate_depth_smail_from_mock_returns_failure_if_alpha_beta_fit_fails():
    """Tests that calibrate_depth_smail_from_mock returns failure when alpha-beta fitting fails."""
    z = np.array([0.1, 0.2, 0.3, 0.4])
    mag = np.array([22.0, 22.1, 22.2, 22.3])
    maglims = np.array([22.0, 22.5, 23.0])

    out = calibrate_depth_smail_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        area_deg2=1.0,
        infer_alpha_beta_from="deep_cut",
    )

    assert out["ok"] is False
    assert out["reason"] == "alpha_beta_fit_failed"
    assert out["alpha_beta_fit"]["ok"] is False


def test_calibrate_depth_smail_from_mock_can_return_false_if_downstream_fit_fails():
    """Tests that calibrate_depth_smail_from_mock reports false ok when a downstream fit fails."""
    z, mag = _make_mock_catalog(n=1500, seed=22)
    maglims = np.array([10.0, 10.5, 11.0])

    out = calibrate_depth_smail_from_mock(
        z_true=z,
        mag=mag,
        maglims=maglims,
        area_deg2=2.0,
        infer_alpha_beta_from="all_selected_at_maglim",
    )

    assert out["alpha_beta_fit"]["ok"] is True
    assert out["z0_of_maglim"]["ok"] is False
    assert out["ok"] is False
