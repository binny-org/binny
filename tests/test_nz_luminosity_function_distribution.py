"""Unit tests for ``binny.nz.luminosity_function_distribution`` module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from binny.nz.luminosity_function_distribution import luminosity_function_distribution


def luminosity_distance_mpc(z: np.ndarray) -> np.ndarray:
    """Return a simple positive luminosity-distance relation."""
    return 100.0 * (1.0 + z)


def volume_weight(z: np.ndarray) -> np.ndarray:
    """Return a simple redshift-dependent volume weight."""
    return 1.0 + z


def constant_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant LF with the magnitude-grid shape."""
    _ = z
    return np.ones_like(m_abs)


class FakeLFKitWithAsCallable:
    """Mock an LFKit object exposing ``_as_callable``."""

    def __init__(self) -> None:
        """Store captured LF inputs."""
        self.captured_m_abs: np.ndarray | None = None
        self.captured_z: np.ndarray | None = None

    def _as_callable(self) -> Any:
        """Return an LF callable."""

        def lf_callable(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
            """Return a constant LF and capture inputs."""
            self.captured_m_abs = m_abs.copy()
            self.captured_z = z.copy()
            return np.ones_like(m_abs)

        return lf_callable


class FakeLFKitWithPhi:
    """Mock an LFKit object exposing ``phi``."""

    def __init__(self) -> None:
        """Store captured LF inputs."""
        self.captured_m_abs: np.ndarray | None = None
        self.captured_z: np.ndarray | None = None

    def phi(self, absolute_mag: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return a constant LF and capture inputs."""
        self.captured_m_abs = absolute_mag.copy()
        self.captured_z = z.copy()
        return np.ones_like(absolute_mag)


def test_constant_lf_and_constant_volume_gives_constant_nz() -> None:
    """Test that constant LF and constant volume give flat n(z)."""
    z = np.linspace(0.1, 1.0, 20)

    nz = luminosity_function_distribution(
        z,
        constant_lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    expected = np.full_like(z, 22.0 - 14.0)

    np.testing.assert_allclose(nz, expected)


def test_constant_lf_with_volume_weight_tracks_volume_weight() -> None:
    """Test that constant LF output follows the volume weight."""
    z = np.linspace(0.1, 1.0, 20)

    nz = luminosity_function_distribution(
        z,
        constant_lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=volume_weight,
        normalize=False,
    )

    expected = (22.0 - 14.0) * volume_weight(z)

    np.testing.assert_allclose(nz, expected)


def test_normalized_output_integrates_to_one() -> None:
    """Test that normalized output integrates to unity."""
    z = np.linspace(0.1, 1.0, 200)

    nz = luminosity_function_distribution(
        z,
        constant_lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=volume_weight,
        normalize=True,
    )

    integral = np.trapezoid(nz, x=z)

    np.testing.assert_allclose(integral, 1.0)


def test_lf_receives_expected_absolute_magnitude_grid() -> None:
    """Test that LF receives M = m - DM(z) without K-correction."""
    z = np.array([0.1, 0.5, 1.0])
    m_bright = 14.0
    m_lim = 22.0
    n_m = 5

    captured: dict[str, np.ndarray] = {}

    def recording_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Capture LF inputs."""
        captured["m_abs"] = m_abs.copy()
        captured["z"] = z_arr.copy()
        return np.ones_like(m_abs)

    luminosity_function_distribution(
        z,
        recording_lf,
        m_bright=m_bright,
        m_lim=m_lim,
        n_m=n_m,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    m_grid = np.linspace(m_bright, m_lim, n_m)
    d_l_pc = luminosity_distance_mpc(z) * 1.0e6
    distance_modulus = 5.0 * np.log10(d_l_pc / 10.0)
    expected_m_abs = m_grid[None, :] - distance_modulus[:, None]

    np.testing.assert_allclose(captured["z"], z)
    np.testing.assert_allclose(captured["m_abs"], expected_m_abs)


def test_k_correction_is_applied_to_absolute_magnitude_grid() -> None:
    """Test that K-correction shifts the absolute-magnitude grid."""
    z = np.array([0.1, 0.5, 1.0])
    m_bright = 14.0
    m_lim = 22.0
    n_m = 5

    def k_correction(z_arr: np.ndarray) -> np.ndarray:
        """Return a simple K-correction."""
        return 0.2 * z_arr

    captured: dict[str, np.ndarray] = {}

    def recording_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Capture LF inputs."""
        _ = z_arr
        captured["m_abs"] = m_abs.copy()
        return np.ones_like(m_abs)

    luminosity_function_distribution(
        z,
        recording_lf,
        m_bright=m_bright,
        m_lim=m_lim,
        n_m=n_m,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        k_correction_fn=k_correction,
        normalize=False,
    )

    m_grid = np.linspace(m_bright, m_lim, n_m)
    d_l_pc = luminosity_distance_mpc(z) * 1.0e6
    distance_modulus = 5.0 * np.log10(d_l_pc / 10.0)
    expected_m_abs = m_grid[None, :] - distance_modulus[:, None] - k_correction(z)[:, None]

    np.testing.assert_allclose(captured["m_abs"], expected_m_abs)


def test_lf_can_use_extra_keyword_arguments() -> None:
    """Test that extra keyword arguments are forwarded to plain LF callables."""
    z = np.linspace(0.1, 1.0, 20)

    def scaled_lf(
        m_abs: np.ndarray,
        z_arr: np.ndarray,
        *,
        amplitude: float,
    ) -> np.ndarray:
        """Return a scaled constant LF."""
        _ = z_arr
        return amplitude * np.ones_like(m_abs)

    nz = luminosity_function_distribution(
        z,
        scaled_lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        amplitude=3.0,
        normalize=False,
    )

    expected = np.full_like(z, 3.0 * (22.0 - 14.0))

    np.testing.assert_allclose(nz, expected)


def test_lfkit_like_object_with_as_callable_gets_2d_redshift_grid() -> None:
    """Test that ``_as_callable`` objects receive broadcastable redshift."""
    z = np.linspace(0.1, 1.0, 20)
    lf = FakeLFKitWithAsCallable()

    nz = luminosity_function_distribution(
        z,
        lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    expected = np.full_like(z, 22.0 - 14.0)

    np.testing.assert_allclose(nz, expected)
    assert lf.captured_m_abs is not None
    assert lf.captured_z is not None
    assert lf.captured_m_abs.shape == (z.size, 64)
    assert lf.captured_z.shape == (z.size, 1)
    np.testing.assert_allclose(lf.captured_z[:, 0], z)


def test_lfkit_like_object_with_phi_gets_2d_redshift_grid() -> None:
    """Test that ``phi`` objects receive broadcastable redshift."""
    z = np.linspace(0.1, 1.0, 20)
    lf = FakeLFKitWithPhi()

    nz = luminosity_function_distribution(
        z,
        lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    expected = np.full_like(z, 22.0 - 14.0)

    np.testing.assert_allclose(nz, expected)
    assert lf.captured_m_abs is not None
    assert lf.captured_z is not None
    assert lf.captured_m_abs.shape == (z.size, 64)
    assert lf.captured_z.shape == (z.size, 1)
    np.testing.assert_allclose(lf.captured_z[:, 0], z)


def test_plain_callable_still_gets_1d_redshift_grid() -> None:
    """Test that plain LF callables still receive one-dimensional redshift."""
    z = np.linspace(0.1, 1.0, 20)
    captured: dict[str, np.ndarray] = {}

    def recording_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Capture LF inputs."""
        captured["m_abs"] = m_abs.copy()
        captured["z"] = z_arr.copy()
        return np.ones_like(m_abs)

    luminosity_function_distribution(
        z,
        recording_lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    assert captured["m_abs"].shape == (z.size, 64)
    assert captured["z"].shape == z.shape
    np.testing.assert_allclose(captured["z"], z)


def test_as_callable_takes_precedence_over_phi() -> None:
    """Test that ``_as_callable`` is preferred when both hooks exist."""

    class FakeLFKitWithBoth:
        """Mock an LFKit object exposing both hooks."""

        def __init__(self) -> None:
            """Store which hook was used."""
            self.used_as_callable = False
            self.used_phi = False

        def _as_callable(self) -> Any:
            """Return an LF callable."""

            def lf_callable(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
                """Mark ``_as_callable`` as used."""
                _ = z
                self.used_as_callable = True
                return np.ones_like(m_abs)

            return lf_callable

        def phi(self, absolute_mag: np.ndarray, z: np.ndarray) -> np.ndarray:
            """Mark ``phi`` as used."""
            _ = z
            self.used_phi = True
            return np.ones_like(absolute_mag)

    z = np.linspace(0.1, 1.0, 20)
    lf = FakeLFKitWithBoth()

    luminosity_function_distribution(
        z,
        lf,
        m_bright=14.0,
        m_lim=22.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc,
        volume_weight_fn=lambda z_arr: np.ones_like(z_arr),
        normalize=False,
    )

    assert lf.used_as_callable
    assert not lf.used_phi


@pytest.mark.parametrize(
    ("z", "message"),
    [
        (np.array(0.5), "z must be a 1D array."),
        (np.array([0.5]), "z must contain at least two points."),
        (np.array([0.0, np.nan]), "z must contain only finite values."),
        (np.array([-0.1, 0.1]), "z must be non-negative."),
    ],
)
def test_invalid_redshift_grid_raises(z: np.ndarray, message: str) -> None:
    """Test that invalid redshift grids raise clear errors."""
    with pytest.raises(ValueError, match=message):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )


def test_invalid_magnitude_grid_settings_raise() -> None:
    """Test that invalid magnitude-grid settings raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    with pytest.raises(ValueError, match="n_m must be at least 2."):
        luminosity_function_distribution(
            z,
            constant_lf,
            n_m=1,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )

    with pytest.raises(ValueError, match="m_lim must be greater than m_bright."):
        luminosity_function_distribution(
            z,
            constant_lf,
            m_bright=22.0,
            m_lim=22.0,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )


def test_bad_luminosity_distance_shape_raises() -> None:
    """Test that bad luminosity-distance shapes raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_distance(z_arr: np.ndarray) -> np.ndarray:
        """Return a bad luminosity-distance shape."""
        return np.ones((z_arr.size, 2))

    with pytest.raises(
        ValueError,
        match=r"luminosity_distance_mpc_fn\(z\) must return shape",
    ):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=bad_distance,
            volume_weight_fn=volume_weight,
        )


def test_non_positive_luminosity_distance_raises() -> None:
    """Test that non-positive luminosity distances raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_distance(z_arr: np.ndarray) -> np.ndarray:
        """Return non-positive luminosity distances."""
        return np.zeros_like(z_arr)

    with pytest.raises(ValueError, match="Luminosity distance must be positive."):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=bad_distance,
            volume_weight_fn=volume_weight,
        )


def test_bad_k_correction_shape_raises() -> None:
    """Test that bad K-correction shapes raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_k_correction(z_arr: np.ndarray) -> np.ndarray:
        """Return a bad K-correction shape."""
        return np.ones((z_arr.size, 2))

    with pytest.raises(ValueError, match=r"k_correction_fn\(z\) must return shape"):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
            k_correction_fn=bad_k_correction,
        )


def test_bad_lf_shape_raises() -> None:
    """Test that bad LF output shapes raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Return a bad LF shape."""
        _ = m_abs
        return np.ones(z_arr.shape)

    with pytest.raises(
        ValueError,
        match=r"lf\(M, z, \.\.\.\) must return an array of shape",
    ):
        luminosity_function_distribution(
            z,
            bad_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )


def test_non_finite_lf_values_raise() -> None:
    """Test that non-finite LF values raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Return non-finite LF values."""
        _ = z_arr
        phi = np.ones_like(m_abs)
        phi[0, 0] = np.nan
        return phi

    with pytest.raises(
        ValueError,
        match=r"lf\(M, z, \.\.\.\) must return only finite values.",
    ):
        luminosity_function_distribution(
            z,
            bad_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )


def test_negative_lf_values_raise() -> None:
    """Test that negative LF values raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_lf(m_abs: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """Return negative LF values."""
        _ = z_arr
        phi = np.ones_like(m_abs)
        phi[0, 0] = -1.0
        return phi

    with pytest.raises(
        ValueError,
        match=r"lf\(M, z, \.\.\.\) must return non-negative values.",
    ):
        luminosity_function_distribution(
            z,
            bad_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=volume_weight,
        )


def test_bad_volume_weight_shape_raises() -> None:
    """Test that bad volume-weight shapes raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_volume_weight(z_arr: np.ndarray) -> np.ndarray:
        """Return a bad volume-weight shape."""
        return np.ones((z_arr.size, 2))

    with pytest.raises(ValueError, match=r"volume_weight_fn\(z\) must return shape"):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=bad_volume_weight,
        )


def test_negative_volume_weight_raises() -> None:
    """Test that negative volume weights raise clear errors."""
    z = np.linspace(0.1, 1.0, 20)

    def bad_volume_weight(z_arr: np.ndarray) -> np.ndarray:
        """Return negative volume weights."""
        weight = np.ones_like(z_arr)
        weight[0] = -1.0
        return weight

    with pytest.raises(
        ValueError,
        match=r"volume_weight_fn\(z\) must return non-negative values.",
    ):
        luminosity_function_distribution(
            z,
            constant_lf,
            luminosity_distance_mpc_fn=luminosity_distance_mpc,
            volume_weight_fn=bad_volume_weight,
        )
