"""Tests for the ``binny.utils.colors.py``."""

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex, to_rgba

from binny.utils.colors import (
    BINNY_CMAP,
    BINNY_CMAP_DISCRETE,
    BINNY_COLORS,
    BINNY_NORM,
    get_binny_colors,
    pastelize,
)


def test_pastelize():
    """Tests that pastelization correctly blends colors with white."""
    assert pastelize("#000000", 0.0) == "#000000"
    assert pastelize("#000000", 1.0) == "#ffffff"
    assert pastelize("#000000", 0.5) == "#808080"
    assert pastelize("#ffffff", 0.5) == "#ffffff"


@pytest.mark.parametrize("amount", [-0.1, 1.1])
def test_pastelize_invalid_amount(amount):
    """Tests that invalid pastelization amounts raise an error."""
    with pytest.raises(ValueError, match="amount"):
        pastelize("#000000", amount)


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_get_binny_colors(n):
    """Tests that the requested number of valid colors is returned."""
    colors = get_binny_colors(n)

    assert len(colors) == n
    assert all(isinstance(color, str) for color in colors)
    assert all(color.startswith("#") for color in colors)


def test_get_binny_colors_viridis():
    """Tests that zero pastelization reproduces the viridis palette."""
    colors = get_binny_colors(5, pastel_amount=0.0)
    viridis = colormaps["viridis"]

    expected = [to_hex(viridis(x)) for x in np.linspace(0.0, 1.0, 5)]

    assert colors == expected


def test_get_binny_colors_alpha():
    """Tests that transparent colors preserve RGB values and set alpha."""
    colors = get_binny_colors(5, alpha=0.5)
    opaque = get_binny_colors(5)

    assert len(colors) == 5

    for color, original in zip(colors, opaque, strict=True):
        assert color == pytest.approx(to_rgba(original, 0.5))


@pytest.mark.parametrize("n", [0, -1])
def test_get_binny_colors_invalid_n(n):
    """Tests that nonpositive color counts raise an error."""
    with pytest.raises(ValueError, match="n must be positive"):
        get_binny_colors(n)


@pytest.mark.parametrize("amount", [-0.1, 1.1])
def test_get_binny_colors_invalid_pastel_amount(amount):
    """Tests that invalid pastel amounts raise an error."""
    with pytest.raises(ValueError, match="pastel_amount"):
        get_binny_colors(5, pastel_amount=amount)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_get_binny_colors_invalid_alpha(alpha):
    """Tests that alpha values outside the unit interval raise an error."""
    with pytest.raises(ValueError, match="alpha"):
        get_binny_colors(5, alpha=alpha)


def test_binny_palette():
    """Tests that the default palette contains five generated colors."""
    assert len(BINNY_COLORS) == 5
    assert BINNY_COLORS == get_binny_colors(5)


def test_binny_colormaps():
    """Tests that continuous and discrete colormaps have the correct colors."""
    assert isinstance(BINNY_CMAP, ListedColormap)
    assert isinstance(BINNY_CMAP_DISCRETE, ListedColormap)

    assert BINNY_CMAP.N == 256
    assert BINNY_CMAP_DISCRETE.N == 5

    for i, color in enumerate(BINNY_COLORS):
        assert to_hex(BINNY_CMAP_DISCRETE(i)) == color


def test_binny_norm():
    """Tests that the discrete normalization maps values to bin indices."""
    assert isinstance(BINNY_NORM, BoundaryNorm)

    np.testing.assert_array_equal(
        BINNY_NORM(np.arange(5) + 0.5),
        np.arange(5),
    )
