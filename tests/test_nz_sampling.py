"""Unit tests for redshift sampling and mock catalog generation."""

import numpy as np
import pytest

from binny.nz.sampling import (
    sample_mock_catalog_from_nz,
    sample_redshifts_from_nz,
)


@pytest.mark.parametrize("n_samples", [1, 2, 100, np.int64(25)])
def test_sample_redshifts_shape_dtype_and_bounds(n_samples):
    """Tests that sampled redshifts have the requested shape, dtype, and bounds."""
    samples = sample_redshifts_from_nz([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], n_samples, seed=42)

    assert samples.shape == (int(n_samples),)
    assert samples.dtype == np.float64
    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0.0) & (samples <= 2.0))


def test_sample_redshifts_uniform_exact_inverse():
    """Tests that uniform sampling exactly reproduces the inverse CDF."""
    seed = 42
    u = np.random.default_rng(seed).random(1000)

    samples = sample_redshifts_from_nz([2.0, 5.0], [1.0, 1.0], u.size, seed=seed)

    np.testing.assert_allclose(samples, 2.0 + 3.0 * u, rtol=1e-13, atol=1e-13)


def test_sample_redshifts_increasing_linear_exact_inverse():
    """Tests that increasing linear densities reproduce the analytic inverse CDF."""
    seed = 42
    u = np.random.default_rng(seed).random(1000)

    samples = sample_redshifts_from_nz([0.0, 1.0], [0.0, 2.0], u.size, seed=seed)

    np.testing.assert_allclose(samples, np.sqrt(u), rtol=1e-12, atol=1e-12)


def test_sample_redshifts_decreasing_linear_exact_inverse():
    """Tests that decreasing linear densities reproduce the analytic inverse CDF."""
    seed = 42
    u = np.random.default_rng(seed).random(1000)

    samples = sample_redshifts_from_nz([0.0, 1.0], [2.0, 0.0], u.size, seed=seed)

    expected = 1.0 - np.sqrt(1.0 - u)
    np.testing.assert_allclose(samples, expected, rtol=1e-11, atol=1e-12)


def test_sample_redshifts_nonuniform_linear_exact_inverse():
    """Tests that a nonuniform grid preserves the analytic linear density."""
    seed = 42
    u = np.random.default_rng(seed).random(1000)

    samples = sample_redshifts_from_nz(
        [0.0, 0.1, 0.4, 1.0],
        [0.0, 0.1, 0.4, 1.0],
        u.size,
        seed=seed,
    )

    np.testing.assert_allclose(samples, np.sqrt(u), rtol=1e-12, atol=1e-12)


def test_sample_redshifts_uniform_statistical_moments():
    """Tests that uniform samples reproduce the expected mean and variance."""
    samples = sample_redshifts_from_nz([0.0, 1.0], [1.0, 1.0], 100_000, seed=42)

    assert np.mean(samples) == pytest.approx(0.5, abs=0.005)
    assert np.var(samples) == pytest.approx(1.0 / 12.0, abs=0.003)


@pytest.mark.parametrize(
    "nz, expected_mean",
    [
        ([0.0, 2.0], 2.0 / 3.0),
        ([2.0, 0.0], 1.0 / 3.0),
        ([0.0, 1.0, 0.0], 1.0),
    ],
)
def test_sample_redshifts_triangular_statistical_mean(nz, expected_mean):
    """Tests that triangular densities reproduce their analytic means."""
    z = np.linspace(0.0, len(nz) - 1, len(nz))
    samples = sample_redshifts_from_nz(z, nz, 50_000, seed=42)

    assert np.mean(samples) == pytest.approx(expected_mean, abs=0.01)


def test_sample_redshifts_nonuniform_uniform_density():
    """Tests that uniform density sampling respects nonuniform interval widths."""
    samples = sample_redshifts_from_nz([0.0, 0.1, 2.0], [1.0, 1.0, 1.0], 100_000, seed=42)

    assert np.mean(samples) == pytest.approx(1.0, abs=0.01)
    assert np.mean(samples < 0.1) == pytest.approx(0.05, abs=0.005)


def test_sample_redshifts_piecewise_interval_probabilities():
    """Tests that interval frequencies follow integrated piecewise-linear density."""
    z = [0.0, 1.0, 3.0]
    nz = [1.0, 3.0, 1.0]

    samples = sample_redshifts_from_nz(z, nz, 100_000, seed=42)

    assert np.mean(samples < 1.0) == pytest.approx(1.0 / 3.0, abs=0.01)


def test_sample_redshifts_internal_zero_density():
    """Tests that intervals with zero integrated density receive no samples."""
    samples = sample_redshifts_from_nz(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        20_000,
        seed=42,
    )

    assert np.all((samples <= 1.0) | (samples >= 3.0))
    assert np.mean(samples <= 1.0) == pytest.approx(0.5, abs=0.02)


def test_sample_redshifts_leading_zero_density():
    """Tests that leading zero-density intervals are skipped."""
    samples = sample_redshifts_from_nz(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0, 1.0],
        1000,
        seed=42,
    )

    assert np.all(samples >= 1.0)


def test_sample_redshifts_trailing_zero_density():
    """Tests that trailing zero-density intervals are skipped."""
    samples = sample_redshifts_from_nz(
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        1000,
        seed=42,
    )

    assert np.all(samples <= 1.0)


def test_sample_redshifts_zero_density_at_endpoint():
    """Tests that a density vanishing at one endpoint remains sampleable."""
    samples = sample_redshifts_from_nz(
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 0.0],
        10_000,
        seed=42,
    )

    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0.0) & (samples <= 1.0))
    assert np.mean(samples) == pytest.approx(0.5, abs=0.01)


def test_sample_redshifts_normalization_invariance():
    """Tests that multiplying the density by a constant preserves samples."""
    z = np.linspace(0.0, 3.0, 101)
    nz = np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2)

    first = sample_redshifts_from_nz(z, nz, 1000, seed=42)
    second = sample_redshifts_from_nz(z, 10.0 * nz, 1000, seed=42)

    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def test_sample_redshifts_very_large_density():
    """Tests that large finite density amplitudes do not overflow normalization."""
    samples = sample_redshifts_from_nz([0.0, 1.0], [1e300, 1e300], 1000, seed=42)

    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_sample_redshifts_very_small_density():
    """Tests that small positive density amplitudes remain sampleable."""
    samples = sample_redshifts_from_nz([0.0, 1.0], [1e-300, 1e-300], 1000, seed=42)

    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_sample_redshifts_seed_reproducibility():
    """Tests that identical seeds produce identical redshift samples."""
    args = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 1000)

    first = sample_redshifts_from_nz(*args, seed=123)
    second = sample_redshifts_from_nz(*args, seed=123)

    np.testing.assert_array_equal(first, second)


def test_sample_redshifts_different_seeds():
    """Tests that different seeds produce different redshift samples."""
    args = ([0.0, 1.0], [1.0, 1.0], 100)

    first = sample_redshifts_from_nz(*args, seed=1)
    second = sample_redshifts_from_nz(*args, seed=2)

    assert not np.array_equal(first, second)


def test_sample_redshifts_seed_and_generator_equivalence():
    """Tests that a seeded generator reproduces direct seed-based sampling."""
    args = ([0.0, 1.0], [1.0, 1.0], 100)

    first = sample_redshifts_from_nz(*args, seed=123)
    second = sample_redshifts_from_nz(*args, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(first, second)


def test_sample_redshifts_generator_state_advances():
    """Tests that successive calls consume the supplied generator state."""
    rng = np.random.default_rng(42)
    args = ([0.0, 1.0], [1.0, 1.0], 100)

    first = sample_redshifts_from_nz(*args, rng=rng)
    second = sample_redshifts_from_nz(*args, rng=rng)

    assert not np.array_equal(first, second)


def test_sample_redshifts_preserves_inputs():
    """Tests that sampling leaves the input grid and density unchanged."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([1.0, 2.0, 1.0])
    original_z = z.copy()
    original_nz = nz.copy()

    sample_redshifts_from_nz(z, nz, 100, seed=42)

    np.testing.assert_array_equal(z, original_z)
    np.testing.assert_array_equal(nz, original_nz)


@pytest.mark.parametrize("n_samples", [0, -1, -100, np.int64(-5)])
def test_sample_redshifts_rejects_nonpositive_counts(n_samples):
    """Tests that nonpositive sample counts raise ValueError."""
    with pytest.raises(ValueError, match="n_samples must be positive"):
        sample_redshifts_from_nz([0.0, 1.0], [1.0, 1.0], n_samples)


@pytest.mark.parametrize("n_samples", [True, False, 1.0, 2.5, "10", None])
def test_sample_redshifts_rejects_invalid_count_types(n_samples):
    """Tests that noninteger sample counts raise TypeError."""
    with pytest.raises(TypeError, match="n_samples must be a positive integer"):
        sample_redshifts_from_nz([0.0, 1.0], [1.0, 1.0], n_samples)


@pytest.mark.parametrize(
    "z, nz, message",
    [
        ([-1.0, 1.0], [1.0, 1.0], "z must be nonnegative"),
        ([0.0, 1.0], [1.0, -1.0], "nz must be nonnegative"),
        ([0.0, 1.0], [0.0, 0.0], "positive integral"),
        ([0.0, 1.0], [1.0], "same shape"),
        ([0.0, 0.0], [1.0, 1.0], "strictly increasing"),
        ([1.0, 0.0], [1.0, 1.0], "strictly increasing"),
        ([0.0], [1.0], "at least two points"),
        ([0.0, np.nan], [1.0, 1.0], "finite"),
        ([0.0, np.inf], [1.0, 1.0], "finite"),
        ([0.0, 1.0], [1.0, np.nan], "finite"),
        ([0.0, 1.0], [1.0, np.inf], "finite"),
        ([[0.0, 1.0]], [[1.0, 1.0]], "1D"),
    ],
)
def test_sample_redshifts_rejects_invalid_distributions(z, nz, message):
    """Tests that invalid grids and densities raise descriptive errors."""
    with pytest.raises(ValueError, match=message):
        sample_redshifts_from_nz(z, nz, 10, seed=42)


def test_sample_redshifts_rejects_seed_and_rng():
    """Tests that seed and rng cannot be supplied simultaneously."""
    with pytest.raises(ValueError, match="either seed or rng"):
        sample_redshifts_from_nz(
            [0.0, 1.0],
            [1.0, 1.0],
            10,
            seed=42,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("bad_rng", [123, "invalid", np.random.RandomState(42)])
def test_sample_redshifts_rejects_invalid_rng(bad_rng):
    """Tests that rng must be a NumPy Generator instance."""
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        sample_redshifts_from_nz([0.0, 1.0], [1.0, 1.0], 10, rng=bad_rng)


def test_sample_mock_catalog_default_keys():
    """Tests that a catalog without an observation model has the expected keys."""
    catalog = sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 100, seed=42)

    assert set(catalog) == {"z", "weight"}
    assert catalog["z"].shape == (100,)
    assert catalog["weight"].shape == (100,)
    assert catalog["z"].dtype == np.float64
    assert catalog["weight"].dtype == np.float64
    np.testing.assert_array_equal(catalog["weight"], np.ones(100))


def test_sample_mock_catalog_matches_direct_sampling():
    """Tests that catalog redshifts match direct sampling with the same seed."""
    z = [0.0, 1.0, 2.0]
    nz = [0.0, 1.0, 0.0]

    samples = sample_redshifts_from_nz(z, nz, 1000, seed=42)
    catalog = sample_mock_catalog_from_nz(z, nz, 1000, seed=42)

    np.testing.assert_array_equal(catalog["z"], samples)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 3.0])
def test_sample_mock_catalog_scalar_weights(weight):
    """Tests that scalar weights are broadcast to all catalog objects."""
    catalog = sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, weights=weight, seed=42)

    np.testing.assert_array_equal(catalog["weight"], np.full(10, weight))


def test_sample_mock_catalog_array_weights():
    """Tests that per-object weights are returned in their original order."""
    weights = np.arange(10, dtype=np.float64)

    catalog = sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, weights=weights, seed=42)

    np.testing.assert_array_equal(catalog["weight"], weights)


def test_sample_mock_catalog_preserves_weight_inputs():
    """Tests that catalog generation does not modify input weight arrays."""
    weights = np.arange(1.0, 11.0)
    original = weights.copy()

    sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, weights=weights, seed=42)

    np.testing.assert_array_equal(weights, original)


def test_sample_mock_catalog_deterministic_observation():
    """Tests that a deterministic observation model produces expected redshifts."""

    def model(z, rng):
        return z + 0.1

    catalog = sample_mock_catalog_from_nz(
        [0.0, 1.0],
        [1.0, 1.0],
        100,
        observation_model=model,
        seed=42,
    )

    assert set(catalog) == {"z", "z_obs", "weight"}
    np.testing.assert_allclose(catalog["z_obs"], catalog["z"] + 0.1)


def test_sample_mock_catalog_identity_observation():
    """Tests that an identity observation model preserves sampled redshifts."""

    def model(z, rng):
        return z

    catalog = sample_mock_catalog_from_nz(
        [0.0, 1.0], [1.0, 1.0], 100, observation_model=model, seed=42
    )

    np.testing.assert_array_equal(catalog["z_obs"], catalog["z"])


def test_sample_mock_catalog_stochastic_observation_reproducibility():
    """Tests that stochastic observations are reproducible with a fixed seed."""

    def model(z, rng):
        return z + rng.normal(0.0, 0.05, size=z.size)

    args = ([0.0, 1.0], [1.0, 1.0], 1000)

    first = sample_mock_catalog_from_nz(*args, observation_model=model, seed=42)
    second = sample_mock_catalog_from_nz(*args, observation_model=model, seed=42)

    np.testing.assert_array_equal(first["z"], second["z"])
    np.testing.assert_array_equal(first["z_obs"], second["z_obs"])
    assert not np.array_equal(first["z"], first["z_obs"])


def test_sample_mock_catalog_observation_uses_shared_rng():
    """Tests that observation noise uses the generator after redshift sampling."""
    seed = 42
    n_samples = 100
    reference_rng = np.random.default_rng(seed)

    expected_z = reference_rng.random(n_samples)
    expected_noise = reference_rng.normal(0.0, 0.1, size=n_samples)

    def model(z, rng):
        return z + rng.normal(0.0, 0.1, size=z.size)

    catalog = sample_mock_catalog_from_nz(
        [0.0, 1.0],
        [1.0, 1.0],
        n_samples,
        observation_model=model,
        seed=seed,
    )

    np.testing.assert_allclose(catalog["z"], expected_z, rtol=0, atol=1e-14)
    np.testing.assert_allclose(catalog["z_obs"], expected_z + expected_noise, rtol=0, atol=1e-14)


def test_sample_mock_catalog_observation_does_not_change_base_samples():
    """Tests that adding an observation model preserves the underlying draws."""

    def model(z, rng):
        return z + rng.normal(0.0, 0.1, size=z.size)

    args = ([0.0, 1.0], [1.0, 1.0], 100)

    plain = sample_mock_catalog_from_nz(*args, seed=42)
    observed = sample_mock_catalog_from_nz(*args, observation_model=model, seed=42)

    np.testing.assert_array_equal(plain["z"], observed["z"])


def test_sample_mock_catalog_observation_cannot_mutate_base_samples():
    """Tests that in-place observation models cannot modify catalog redshifts."""

    def model(z, rng):
        z[:] = 0.5
        return z

    catalog = sample_mock_catalog_from_nz(
        [0.0, 1.0],
        [1.0, 1.0],
        100,
        observation_model=model,
        seed=42,
    )

    np.testing.assert_array_equal(catalog["z_obs"], np.full(100, 0.5))
    assert not np.all(catalog["z"] == 0.5)


def test_sample_mock_catalog_allows_negative_observed_redshifts():
    """Tests that observation models may produce finite negative redshifts."""

    def model(z, rng):
        return z - 2.0

    catalog = sample_mock_catalog_from_nz(
        [0.0, 1.0], [1.0, 1.0], 100, observation_model=model, seed=42
    )

    assert np.all(catalog["z_obs"] < 0.0)
    assert np.all(catalog["z"] >= 0.0)


def test_sample_mock_catalog_observation_model_receives_generator():
    """Tests that the observation callable receives a NumPy random generator."""
    received = []

    def model(z, rng):
        received.append(isinstance(rng, np.random.Generator))
        return z

    sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, observation_model=model, seed=42)

    assert received == [True]


def test_sample_mock_catalog_generator_reproducibility():
    """Tests that existing generators reproduce the same complete catalog."""

    def model(z, rng):
        return z + rng.normal(0.0, 0.1, size=z.size)

    args = ([0.0, 1.0], [1.0, 1.0], 100)

    first = sample_mock_catalog_from_nz(
        *args, observation_model=model, rng=np.random.default_rng(123)
    )
    second = sample_mock_catalog_from_nz(
        *args, observation_model=model, rng=np.random.default_rng(123)
    )

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("bad", [123, "gaussian", [1, 2]])
def test_sample_mock_catalog_rejects_noncallable_model(bad):
    """Tests that observation models must be callable."""
    with pytest.raises(TypeError, match="observation_model must be callable"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, observation_model=bad)


@pytest.mark.parametrize(
    "model",
    [
        lambda z, rng: np.zeros((z.size, 1)),
        lambda z, rng: np.zeros(z.size - 1),
        lambda z, rng: 0.5,
    ],
)
def test_sample_mock_catalog_rejects_observation_shape(model):
    """Tests that observation models return exactly one redshift per object."""
    with pytest.raises(ValueError, match="one redshift per sampled object"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, observation_model=model)


@pytest.mark.parametrize(
    "model",
    [
        lambda z, rng: np.full(z.size, np.nan),
        lambda z, rng: np.full(z.size, np.inf),
        lambda z, rng: np.full(z.size, -np.inf),
    ],
)
def test_sample_mock_catalog_rejects_nonfinite_observations(model):
    """Tests that observation models cannot return nonfinite redshifts."""
    with pytest.raises(ValueError, match="finite redshifts"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, observation_model=model)


def test_sample_mock_catalog_rejects_nonnumeric_observations():
    """Tests that observation models must return numeric values."""

    def model(z, rng):
        return ["invalid"] * z.size

    with pytest.raises(ValueError, match="numeric redshifts"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, observation_model=model)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 2.0],
        np.ones((10, 1)),
        np.ones((2, 5)),
        -1.0,
        np.nan,
        np.inf,
        [1.0] * 9 + [-1.0],
        [1.0] * 9 + [np.nan],
        "invalid",
    ],
)
def test_sample_mock_catalog_rejects_invalid_weights(weights):
    """Tests that invalid catalog weights raise ValueError."""
    with pytest.raises(ValueError, match="weights"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], 10, weights=weights)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
def test_sample_mock_catalog_rejects_invalid_sample_counts(bad):
    """Tests that catalog generation validates the requested object count."""
    error = ValueError if isinstance(bad, int) and not isinstance(bad, bool) else TypeError

    with pytest.raises(error, match="n_samples"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, 1.0], bad)


def test_sample_mock_catalog_rejects_seed_and_rng():
    """Tests that catalog generation rejects simultaneous seed and rng."""
    with pytest.raises(ValueError, match="either seed or rng"):
        sample_mock_catalog_from_nz(
            [0.0, 1.0],
            [1.0, 1.0],
            10,
            seed=42,
            rng=np.random.default_rng(42),
        )


def test_sample_mock_catalog_rejects_invalid_distribution():
    """Tests that catalog generation propagates distribution validation errors."""
    with pytest.raises(ValueError, match="nz must be nonnegative"):
        sample_mock_catalog_from_nz([0.0, 1.0], [1.0, -1.0], 10, seed=42)
