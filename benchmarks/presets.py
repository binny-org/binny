import os

import numpy as np
import yaml


class Presets:

    SUPPORTED_FORECAST_YEARS = {"1", "4", "7", "10"}
    SUPPORTED_CORRELATION_PAIR_TYPES = {"srd", "all"}

    def __init__(self,
                 forecast_year,
                 redshift_range=None,
                 correlation_pair_type="srd",

                 ):
        """
        Initialize the Presets object.

        Args:
            redshift_range (numpy.ndarray): The redshift range for the analysis.
            forecast_year (str): The forecast year.
            correlation_pair_type (str): The type of correlation correlations to use.
        """

        if redshift_range is None:
            self.redshift_range = np.linspace(0.0, 3.5, 500)
            print("No redshift range provided. Using default: 0.0 <= z <= 3.5 with 500 points.")
        else:
            self.redshift_range = redshift_range

        self.forecast_year = self.validate_input(forecast_year, self.SUPPORTED_FORECAST_YEARS)
        self.correlation_pair_type = self.validate_input(correlation_pair_type, self.SUPPORTED_CORRELATION_PAIR_TYPES)

        # Directory containing this file (benchmarks/)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        lsst_desc_path = os.path.join(script_dir, "lsst_desc_specs.yaml")
        self.base_dir = script_dir

        if not os.path.exists(lsst_desc_path):
            raise FileNotFoundError(f"File not found: {lsst_desc_path}")

        with open(lsst_desc_path, "r") as f:
            lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.lens_params = lsst_desc_parameters["lens_sample"][self.forecast_year]
        self.source_params = lsst_desc_parameters["source_sample"][self.forecast_year]
        self.lens_type = "lens_sample"
        self.source_type = "source_sample"
        # Define limiting magnitudes (float and string versions)
        self.limiting_magnitudes_float = np.round(np.arange(21.5, 26.5 + 0.1, 0.1), 1)
        self.limiting_magnitudes_str = [f"{m:.1f}" for m in self.limiting_magnitudes_float]

    def validate_input(self, input_value, supported_options):
        """
        Validate the given input value against a set of supported options.

        Args:
            input_value (str): The value to be validated.
            supported_options (set or list): The set or list of supported options.

        Returns:
            str: The validated input value.

        Raises:
            ValueError: If the input value is not in the set of supported options.
        """
        if input_value not in supported_options:
            raise ValueError(f"'{input_value}' must be one of {supported_options}.")
        return input_value
