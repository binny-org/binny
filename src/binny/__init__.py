"""Public Binny API."""

from binny.nz import (
    available_models,
    get_model,
    nz_model,
    sample_mock_catalog_from_nz,
    sample_redshifts_from_nz,
)
from binny.surveys.survey_presets import (
    build_survey_tomography,
    list_survey_configs,
    load_survey_config,
    show_survey_config,
)
from binny.tomography import (
    Tomography,
    build_predefined_tomography,
    build_tomography,
    build_tomography_from_spec,
    load_predefined_bins,
)
from binny.utils.colors import (
    BINNY_CMAP,
    BINNY_CMAP_DISCRETE,
    BINNY_COLORS,
    BINNY_NORM,
    get_binny_colors,
    pastelize,
)

__all__ = [
    "Tomography",
    "build_tomography",
    "build_tomography_from_spec",
    "build_survey_tomography",
    "build_predefined_tomography",
    "load_predefined_bins",
    "available_models",
    "get_model",
    "nz_model",
    "sample_redshifts_from_nz",
    "sample_mock_catalog_from_nz",
    "list_survey_configs",
    "load_survey_config",
    "show_survey_config",
    "BINNY_COLORS",
    "BINNY_CMAP",
    "BINNY_CMAP_DISCRETE",
    "BINNY_NORM",
    "get_binny_colors",
    "pastelize",
]
