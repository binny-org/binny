"""Generate benchmark datasets for LSST galaxy samples.

This script creates reference datasets for various redshift ranges,
grid resolutions, and forecast years (Y1 and Y10) using the
``LSSTGalaxySample`` class from the ``lsst_galaxy_sample.py``. The
generated datasets include source and lens redshift distributions,
tomographic bin edges, bin centers, and effective number densities.
LSST survey specs are stored in the ``lsst_desc_specs.yaml`` file.

The LSST code is the source of truth for these datasets, and it
was written by Niko Sarcevic (GitHub @nikosarcevic) for the
LSST DESC collaboration.

The data is generated and stored in tests/reference/data/ directory
under subdirectories named according to the redshift range and grid
resolution (e.g., "default__default" for the default range and
resolution). Each dataset is saved as a NumPy .npy file for easy
loading and comparison in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lsst_galaxy_sample import LSSTGalaxySample
from presets import Presets

# === Define redshift ranges and grid resolutions ===
range_types = {
    "narrow": (0.0, 2.0),
    "default": (0.0, 3.5),
    "wide": (0.0, 5.0),
}

grid_resolutions = {
    "coarse": 200,
    "default": 500,
    "fine": 1000,
    "superfine": 2000,
}

forecast_years = ["1", "10"]

# === Setup base output directory (repo-root/tests/reference/data) ===
REPO_ROOT = Path(__file__).resolve().parents[1]
base_dir = REPO_ROOT / "tests" / "reference" / "data"
base_dir.mkdir(parents=True, exist_ok=True)

# === Loop over all range/grid combinations and forecast years ===
for range_name, (zmin, zmax) in range_types.items():
    for resolution_name, npoints in grid_resolutions.items():
        redshift_grid = np.linspace(zmin, zmax, npoints)
        grid_tag = f"{range_name}__{resolution_name}"

        save_dir = base_dir / grid_tag
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating data for grid: {grid_tag}")

        for year in forecast_years:
            print(f"  - Forecast Year {year}")

            presets = Presets(forecast_year=year, redshift_range=redshift_grid)
            init = LSSTGalaxySample(presets)

            source_nz = init.source_sample(normalized=True, save_file=False)
            lens_nz = init.lens_sample(normalized=True, save_file=False)

            source_bins = init.source_bins(normalized=True, save_file=False)
            lens_bins = init.lens_bins(normalized=True, save_file=False)

            source_bin_centers = init.compute_tomo_bin_centers(
                source_bins, decimal_places=4
            )
            lens_bin_centers = init.compute_tomo_bin_centers(
                lens_bins, decimal_places=4
            )

            lens_neff = init.get_n_eff_clustering(save_file=False)
            source_neff = init.get_n_eff_lensing(save_file=False)

            lens_neff_frac = init.get_n_eff_frac_clustering(save_file=False)
            source_neff_frac = init.get_n_eff_frac_lensing(save_file=False)

            sample_data = {
                "source_nz": source_nz,
                "lens_nz": lens_nz,
                "source_bins": source_bins,
                "lens_bins": lens_bins,
                "source_bin_centers": source_bin_centers,
                "lens_bin_centers": lens_bin_centers,
                "lens_neff_per_bin": lens_neff,
                "source_neff_per_bin": source_neff,
                "lens_neff_frac_per_bin": lens_neff_frac,
                "source_neff_frac_per_bin": source_neff_frac,
                "redshift_range": redshift_grid,
            }

            for key, value in sample_data.items():
                filepath = save_dir / f"{key}_Y{year}.npy"
                np.save(filepath, value)

print(f"\nAll reference datasets saved to: {base_dir}")
