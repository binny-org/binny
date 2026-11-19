"""Redshift distributions and mock catalog generation."""

from binny.nz.models import (
    gamma_distribution,
    gaussian_distribution,
    gaussian_mixture_distribution,
    lognormal_distribution,
    luminosity_function_distribution,
    schechter_like_distribution,
    shifted_smail_distribution,
    skew_normal_distribution,
    smail_like_distribution,
    student_t_distribution,
    tabulated_distribution,
    tophat_distribution,
)
from binny.nz.registry import available_models, get_model, nz_model
from binny.nz.sampling import (
    sample_mock_catalog_from_nz,
    sample_redshifts_from_nz,
)
from binny.utils.io import load_nz

__all__ = [
    "available_models",
    "gamma_distribution",
    "gaussian_distribution",
    "gaussian_mixture_distribution",
    "get_model",
    "load_nz",
    "lognormal_distribution",
    "luminosity_function_distribution",
    "nz_model",
    "sample_mock_catalog_from_nz",
    "sample_redshifts_from_nz",
    "schechter_like_distribution",
    "shifted_smail_distribution",
    "skew_normal_distribution",
    "smail_like_distribution",
    "student_t_distribution",
    "tabulated_distribution",
    "tophat_distribution",
]
