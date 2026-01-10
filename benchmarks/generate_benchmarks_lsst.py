import os

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

forecast_years = ["1", "4", "7", "10"]

# === Setup base output directory ===
base_dir = os.path.abspath(os.path.join("benchmarks", "data"))
os.makedirs(base_dir, exist_ok=True)

# === Loop over all range/grid combinations and forecast years ===
for range_name, (zmin, zmax) in range_types.items():
    for resolution_name, npoints in grid_resolutions.items():
        redshift_grid = np.linspace(zmin, zmax, npoints)
        grid_tag = f"{range_name}__{resolution_name}"

        save_dir = os.path.join(base_dir, grid_tag)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nGenerating data for grid: {grid_tag}")

        for year in forecast_years:
            print(f"  - Forecast Year {year}")

            # Initialize with current config
            presets = Presets(forecast_year=year, redshift_range=redshift_grid)
            init = LSSTGalaxySample(presets)

            # === Step 1: Compute core inputs once ===
            source_nz = init.source_sample(normalized=True, save_file=False)
            lens_nz = init.lens_sample(normalized=True, save_file=False)

            # === Step 2: Compute everything else using the cached inputs ===
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

            # === Step 3: Save once, outside ===
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
                filename = f"{key}_Y{year}.npy"
                filepath = os.path.join(save_dir, filename)
                np.save(filepath, value)

print(f"\n All benchmark datasets saved to: {base_dir}")
