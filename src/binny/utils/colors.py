"""Define the Binny color palette and Matplotlib colormaps."""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    to_hex,
    to_rgb,
    to_rgba,
)

__all__ = [
    "BINNY_COLORS",
    "BINNY_CMAP",
    "BINNY_CMAP_DISCRETE",
    "BINNY_NORM",
    "get_binny_colors",
    "pastelize",
]

PASTEL_AMOUNT = 0.25
N_COLORS = 5

_BASE_CMAP = colormaps["viridis"]


def pastelize(color: str, amount: float) -> str:
    """Blend a color with white to produce a pastel shade.

    Args:
        color: Matplotlib-compatible color.
        amount: Fraction of white to mix in, between 0 and 1.

    Returns:
        Hexadecimal color string.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must lie between 0 and 1.")

    rgb = np.asarray(to_rgb(color))
    pastel = (1.0 - amount) * rgb + amount * np.ones(3)
    return to_hex(pastel)


def get_binny_colors(
    n: int,
    *,
    pastel_amount: float = PASTEL_AMOUNT,
    alpha: float = 1.0,
) -> list[str] | list[tuple[float, float, float, float]]:
    """Sample colors from the Binny palette.

    Args:
        n: Number of colors.
        pastel_amount: Fraction of white mixed into each color.
        alpha: Opacity between 0 and 1.

    Returns:
        Hexadecimal colors if alpha is 1, otherwise RGBA colors.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    if not 0.0 <= pastel_amount <= 1.0:
        raise ValueError("pastel_amount must lie between 0 and 1.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie between 0 and 1.")

    colors = [pastelize(to_hex(_BASE_CMAP(x)), pastel_amount) for x in np.linspace(0.0, 1.0, n)]

    if alpha == 1.0:
        return colors

    return [to_rgba(color, alpha) for color in colors]


BINNY_COLORS = get_binny_colors(N_COLORS)

BINNY_CMAP = ListedColormap(
    get_binny_colors(256),
    name="binny_colors",
)

BINNY_CMAP_DISCRETE = ListedColormap(
    BINNY_COLORS,
    name="binny_colors_discrete",
)

BINNY_NORM = BoundaryNorm(
    np.arange(len(BINNY_COLORS) + 1),
    BINNY_CMAP_DISCRETE.N,
)
