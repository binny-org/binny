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
    "matplotlib.sphinxext.plot_directive",
]

# General plot settings
plot_rcparams = {
    # tick style
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    # global font sizes
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
    # tick label sizes
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    # nice defaults
    "axes.linewidth": 2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
}

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
}

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
        "source_repository": "https://github.com/binny-org/binny/",
        "source_branch": "main",
        "source_directory": "docs/",
        "light_logo": "assets/logo.png",
        "dark_logo": "assets/logo.png",
        "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/binny-org/binny",
                "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                    2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                    2.2.82A7.65 7.65 0 0 1 8 3.87c.68 0 1.36.09 2 .26
                    1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                    .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65
                    3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01
                    2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16
                    8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
                "class": "",
            },
        ],
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
