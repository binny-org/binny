# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "binny"
copyright = "2026, Nikolina Šarčević"
author = "Nikolina Šarčević"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
]

autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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

# -- Sphinx Multiversion --------------------------------------------------
# https://sphinx-contrib.github.io/multiversion/main/configuration.html

smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = "main"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = "<span>#</span>"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
