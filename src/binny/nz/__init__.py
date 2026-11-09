"""Redshift-distribution models, calibration, and selection."""

from binny.nz.calibration import (
    calibrate_depth_smail_from_mock,
    eval_ngal_of_maglim,
    eval_z0_of_maglim,
    fit_ngal_of_maglim_from_mock,
    fit_smail_params_from_mock,
    fit_z0_of_maglim_from_mock,
)
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
from binny.nz.psf_selection import (
    calibrate_psf_depth_from_mock,
    resolution_factor,
    selected_number_density_from_mock,
    shear_selection_weight,
    weighted_nz_from_mock,
)
from binny.nz.registry import available_models, get_model, nz_model

__all__ = [
    "available_models",
    "get_model",
    "nz_model",
    "smail_like_distribution",
    "gaussian_distribution",
    "gaussian_mixture_distribution",
    "gamma_distribution",
    "schechter_like_distribution",
    "lognormal_distribution",
    "tophat_distribution",
    "shifted_smail_distribution",
    "skew_normal_distribution",
    "student_t_distribution",
    "tabulated_distribution",
    "luminosity_function_distribution",
    "fit_smail_params_from_mock",
    "fit_z0_of_maglim_from_mock",
    "eval_z0_of_maglim",
    "fit_ngal_of_maglim_from_mock",
    "eval_ngal_of_maglim",
    "calibrate_depth_smail_from_mock",
    "resolution_factor",
    "shear_selection_weight",
    "weighted_nz_from_mock",
    "selected_number_density_from_mock",
    "calibrate_psf_depth_from_mock",
]
