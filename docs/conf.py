"""Sphinx configuration for the Binny documentation."""

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import warnings
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------
import cmasher as cmr

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Binny theme colors
# -----------------------------------------------------------------------------
DEFAULT_CMAP = "viridis"
DEFAULT_CMAP_RANGE = (0.0, 1.0)


warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"colorspacious\.comparison",
)


def get_binny_theme_colors(
    cmap: str = DEFAULT_CMAP,
    cmap_range: tuple[float, float] = DEFAULT_CMAP_RANGE,
) -> tuple[str, str, str, str]:
    """Return the four default Binny theme colors as hex strings.

    Colors are sampled from the colormap in the order:

        purple → blue → green → yellow

    Args:
        cmap:
            Name of the colormap.
        cmap_range:
            Fractional range of the colormap to sample.

    Returns
    -------
    tuple[str, str, str, str]
        Hex colors (purple, blue, green, yellow).
    """

    purple, blue, green, yellow = cmr.take_cmap_colors(
        cmap,
        4,
        cmap_range=cmap_range,
        return_fmt="hex",
    )

    return purple, blue, green, yellow


BINNY_PURPLE, BINNY_BLUE, BINNY_GREEN, BINNY_YELLOW = get_binny_theme_colors()

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "Binny"
copyright = "2026, Nikolina Šarčević, Matthijs van der Wild"
author = "Nikolina Šarčević"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinx_multiversion",
    "sphinx_copybutton",
]

autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "binny.core.rst"]

# -----------------------------------------------------------------------------
# Doctest configuration
# -----------------------------------------------------------------------------
doctest_global_setup = r"""
import numpy as np
np.set_printoptions(precision=12, suppress=True)
"""

# -----------------------------------------------------------------------------
# Copybutton configuration
# -----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False

# -----------------------------------------------------------------------------
# Sidebar layout
# -----------------------------------------------------------------------------
html_sidebar = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/variant-selector.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}

# -----------------------------------------------------------------------------
# Sphinx Multiversion
# -----------------------------------------------------------------------------
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = "main"

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = "furo"
html_favicon = "_static/assets/favicon.png"
html_permalinks_icon = "<span>#</span>"

if html_theme == "furo":
    html_theme_options = {
        "light_logo": "assets/logo.png",
        "dark_logo": "assets/logo.png",
        "light_css_variables": {
            "color-brand-primary": BINNY_PURPLE,
            "color-brand-content": BINNY_BLUE,
            "color-link": BINNY_GREEN,
            "color-link--hover": BINNY_BLUE,
            "color-link--visited": BINNY_YELLOW,
        },
        "dark_css_variables": {
            "color-brand-primary": BINNY_PURPLE,
            "color-brand-content": BINNY_BLUE,
            "color-link": BINNY_GREEN,
            "color-link--hover": BINNY_BLUE,
            "color-link--visited": BINNY_YELLOW,
        },
    }

html_static_path = ["_static"]

html_css_files = [
    "binny.css",
]
