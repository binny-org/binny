#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github/nikosarcevic
#  ----------

import os

import numpy as np
import pandas
import yaml
from numpy import exp
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import erf

from presets import Presets


class LSSTGalaxySample:
    """
    Handles the generation and processing of redshift distributions and tomographic binning
    for LSST DESC galaxy samples (source and lens) for different forecast years.
    Combines functionality for generating Smail-type distributions and slicing them into bins.
    """

    def __init__(self, presets: Presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")

        self.redshift_range = presets.redshift_range
        self.forecast_year = presets.forecast_year

        # Resolve path relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lsst_desc_path = os.path.join(script_dir, "lsst_desc_specs.yaml")

        with open(lsst_desc_path, "r") as f:
            lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.lens_params = lsst_desc_parameters["lens_sample"][self.forecast_year]
        self.source_params = lsst_desc_parameters["source_sample"][self.forecast_year]
        self.lens_type = "lens_sample"
        self.source_type = "source_sample"

    def lens_sample(self, normalized=True, save_file=False, file_format="npy"):
        """
        Generates the lens sample redshift distribution.

        Arguments:
            normalized (bool): Normalize the redshift distribution (default: True).
            save_file (bool): Save the output file (default: True).
            file_format (str): Output file format ('npy', 'csv').

        Returns:
            np.array: Normalized p(z), suitable for CCL.
        """
        # Compute the Smail-type redshift distribution (not yet normalized)
        p_z = self.smail_type_distribution(self.redshift_range,
                                           self.lens_params["z_0"],
                                           self.lens_params["alpha"],
                                           self.lens_params["gamma"])

        # Normalize the distribution if requested (ALWAYS needed for CCL)
        if normalized:
            p_z = self.normalize_distribution(p_z, method="trapezoid")

        # Save the data if requested
        if save_file:
            self.save_to_file("lens_sample_nz",
                              {"redshift": self.redshift_range,
                               "distribution": p_z}, file_format)

        return p_z

    def source_sample(self, normalized=True, save_file=False, file_format="npy"):
        """
        Generates the source sample redshift distribution.

        Arguments:
            normalized (bool): Normalize the redshift distribution (default: True).
            save_file (bool): Save the output file (default: True).
            file_format (str): Output file format ('npy', 'csv').

        Returns:
            np.array: Normalized p(z), suitable for CCL.
        """
        # Compute the Smail-type redshift distribution (not yet normalized)
        p_z = self.smail_type_distribution(self.redshift_range,
                                           self.source_params["z_0"],
                                           self.source_params["alpha"],
                                           self.source_params["gamma"])

        # Normalize the distribution if requested (ALWAYS needed for CCL)
        if normalized:
            p_z = self.normalize_distribution(p_z, method="trapezoid")

        # Save the data if requested
        if save_file:
            self.save_to_file("source_sample_nz",
                              {"redshift": self.redshift_range, "distribution": p_z},
                              file_format)

        return p_z

    def lens_bins(self,
                  normalized=True,
                  save_file=False,
                  file_format='npy'):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins.
        In the LSST DESC case, lenses are split into 5 tomographic bins (year 1 forecast) or 10
        tomographic bins (year 10). Binning is performed in such a way that the bins are spaced
        by 0.1 in photo-z between 0.2 ≤ z ≤ 1.2 for Y10, and 5 bins spaced by 0.2 in photo-z in
        the same redshift range.
        Variance is 0.03 for both forecast years while z_bias is zero.
        For more information about the redshift distributions and binning,
        consult the LSST DESC Science Requirements Document (SRD) https://arxiv.org/abs/1809.01669,
        Appendix D.
        ----------
        Arguments:
            normalized: bool
                normalize the redshift distribution (defaults to True).
            save_file: bool
                option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distributions for each bin. The bin headers are the
                dictionary keys.
            file_format: str (defaults to 'npy')
                format of the output file. Accepted values are 'csv' and 'npy'.
        Returns: dictionary
                A lens galaxy sample, appropriately binned. Depending on the forecast year
                chosen while initialising the class, it will output a lens sample for year 1
                (5 bins) or lens galaxy sample for year 10 (10 bins).
        """
        # Generate the redshift distribution for the lens sample
        lens_dndz = self.lens_sample(normalized=normalized)
        # Define the bin edges
        bin_edges = self.get_tomo_bin_edges("lens")

        # Get the bias and variance values for each bin
        z_bias, z_variance = self.get_biases("lens")

        # Create a dictionary of the redshift distributions for each bin
        redshift_distribution_dict = self.create_redshift_bins(lens_dndz,
                                                               bin_edges,
                                                               z_bias,
                                                               z_variance)

        # Normalise the distributions
        if normalized:
            redshift_distribution_dict = self.normalize_distribution(redshift_distribution_dict,
                                                                     method="trapezoid")

        combined_data = {'redshift_range': self.redshift_range,
                         'bins': redshift_distribution_dict}

        # Save the distributions to a file
        if save_file:
            self.save_to_file("lens_bins", combined_data, file_format)

        return redshift_distribution_dict

    def source_bins(self, normalized=True, save_file=False, file_format='npy'):
        """split the initial redshift distribution of source galaxies into tomographic bins.
        LSST DESC case, sources are split into 5 tomographic bins (year 1 and year 10 forecast).
        Each bin has equal number of galaxies. Variance is 0.05 for both forecast years while z_bias is zero.
        For more information about the redshift distributions and binning,
        consult the LSST DESC Science Requirements Document (SRD) https://arxiv.org/abs/1809.01669,
        Appendix D.
        ----------
        Arguments:
            normalized (bool): normalize the redshift distribution (defaults to True).
            save_file (bool): option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distributions for each bin. The bin headers are the
                dictionary keys.
            file_format (str): format of the output file (defaults to 'npy').
                Accepted values are 'csv' and 'npy'.
        Returns:
            A source galaxy sample (dictionary), appropriately binned."""
        # Generate the redshift distribution for the source sample
        source_dndz = self.source_sample(normalized=normalized)
        # Get the bin edges as redshift values directly
        bin_edges = self.get_tomo_bin_edges("source")

        # Get the bias and variance values for each bin
        z_bias, z_variance = self.get_biases("source")

        # Create a dictionary of the redshift distributions for each bin
        redshift_distribution_dict = self.create_redshift_bins(source_dndz,
                                                               bin_edges,
                                                               z_bias,
                                                               z_variance)

        # Normalise the distributions
        if normalized:
            redshift_distribution_dict = self.normalize_distribution(redshift_distribution_dict,
                                                                     method="trapezoid")

            # Create a combined dictionary
        combined_data = {'redshift_range': self.redshift_range,
                         'bins': redshift_distribution_dict}

        # Save the data
        if save_file:
            self.save_to_file("source_bins", combined_data, file_format)

        return redshift_distribution_dict

    def lens_bin_centers(self, decimal_places=2, save_file=False, file_format="npy"):
        """
        Compute the lens bin centers for the LSST DESC forecast year.

        Arguments:
            decimal_places (int): Number of decimal places to round the bin centers.
            save_file (bool): Option to save the output as a .csv (default: True).
            file_format (str): Format of the output file ('npy', 'csv').

        Returns:
            bin_centers (dict): Dictionary of bin centers with bin indices as keys.
        """
        lens_bins = self.lens_bins(normalized=True, save_file=False)
        bin_centers = self.compute_tomo_bin_centers(lens_bins, decimal_places=decimal_places)
        if save_file:
            self.save_to_file("lens_bin_centers", bin_centers, file_format)

        return bin_centers

    def source_bin_centers(self, decimal_places=2, save_file=False, file_format="npy"):
        """
        Compute the source bin centers for the LSST DESC forecast year.

        Arguments:
            decimal_places (int): Number of decimal places to round the bin centers.
            save_file (bool): Option to save the output as a .csv (default: True).
            file_format (str): Format of the output file ('npy', 'csv').

        Returns:
            bin_centers (dict): Dictionary of bin centers with bin indices as keys.
        """
        source_bins = self.source_bins(normalized=True, save_file=False)
        bin_centers = self.compute_tomo_bin_centers(source_bins, decimal_places=decimal_places)
        if save_file:
            self.save_to_file("source_bin_centers", bin_centers, file_format)
        return bin_centers

    def smail_type_distribution(self,
                                redshift_range,
                                pivot_redshift,
                                alpha,
                                gamma
                                ):

        """
        Generate the LSST DESC SRD parametric redshift distribution (Smail-type).
        For details check LSST DESC SRD paper https://arxiv.org/abs/1809.01669, equation 5.
        The redshift distribution parametrisation is a Smail type of the form
        N(z) = (z / z0) ^ gamma * exp[- (z / z0) ^ alpha],
        where z is redshift, z0 is pivot redshift, and alpha and gamma are power law indices.
        ----------
        Arguments:
            redshift_range: array
                redshift range
            pivot_redshift: float
                pivot redshift
            alpha: float
                power law index in the exponent
            gamma: float
                power law index in the prefactor
        Returns:
            redshift_distribution: array
                A Smail-type redshift distribution over a range of redshifts.
                """

        pz = [(z / pivot_redshift) ** gamma * exp(-(z / pivot_redshift) ** alpha) for z in redshift_range]

        return np.array(pz)

    def true_redshift_distribution(self, redshift_distribution, bin_min, bin_max, sigma_z, z_bias):
        """Compute the true (photometric) redshift distribution of a galaxy sample.

        The true distribution is obtained by applying the probability distribution p(z_phot | z),
        which accounts for photometric errors, to the overall galaxy redshift distribution.

        This follows Ma, Hu & Huterer 2018 (https://arxiv.org/abs/astro-ph/0506614, Eq. 6).

        Arguments:
            redshift_distribution (array): The intrinsic redshift distribution n(z).
            bin_min (float): Lower bound of the redshift bin.
            bin_max (float): Upper bound of the redshift bin.
            sigma_z (float): Photometric redshift scatter (variance).
            z_bias (float): Photometric redshift bias.

        Returns:
            p_z_photometric (array): The convolved redshift distribution after including photometric errors.
        """
        # Compute photometric scatter as a function of redshift
        scatter = np.maximum(sigma_z * (1 + self.redshift_range), 1e-10)  # Avoid division by zero

        # Compute integration limits
        upper_limit = (bin_max - self.redshift_range + z_bias) / (np.sqrt(2) * scatter)
        lower_limit = (bin_min - self.redshift_range + z_bias) / (np.sqrt(2) * scatter)

        # Compute the photometric redshift distribution
        p_z_photometric = 0.5 * np.array(redshift_distribution) * (erf(upper_limit) - erf(lower_limit))

        return p_z_photometric

    def compute_cumulative_distribution(self, redshift_range, redshift_distribution):
        """
        Computes the cumulative distribution and total number of galaxies.

        Arguments:
            redshift_range: ndarray
                Array of redshift values.
            redshift_distribution: ndarray
                Redshift distribution over the given range.

        Returns:
            cumulative_distribution: ndarray
                The cumulative distribution of the redshift values.
            total_galaxies: float
                The total number of galaxies (area under the distribution).
        """
        cumulative_distribution = cumulative_trapezoid(redshift_distribution, redshift_range, initial=0)
        total_galaxies = cumulative_distribution[-1]
        return cumulative_distribution, total_galaxies

    def compute_equal_number_bounds(self, redshift_range, redshift_distribution, n_bins):
        """
        Determines the redshift values that divide the distribution into bins
        with an equal number of galaxies.

        Arguments:
            redshift_range (array): an array of redshift values
            redshift_distribution (array): the corresponding redshift distribution defined over redshift_range
            n_bins (int): the number of tomographic bins

        Returns:
            An array of redshift values that are the boundaries of the bins.
        """
        redshift_range = np.asarray(redshift_range)
        redshift_distribution = np.asarray(redshift_distribution)

        # Calculate the cumulative distribution
        cumulative_distribution, total_galaxies = self.compute_cumulative_distribution(redshift_range,
                                                                                       redshift_distribution)

        # Find the bin edges
        bin_edges = []
        for i in range(1, n_bins):
            fraction = i / n_bins * total_galaxies
            # Find the redshift value where the cumulative distribution crosses this fraction
            bin_edge = np.interp(fraction, cumulative_distribution, redshift_range)
            bin_edges.append(bin_edge)

        return [redshift_range[0]] + bin_edges + [redshift_range[-1]]

    def get_biases(self, sample_type):
        """
        Get the z_bias and z_variance values for the given galaxy sample type.
        Parameters
        ----------
        sample_type: str
            the type of the galaxy sample (lens or source)

        Returns
        -------
        z_bias: array
            the bias values for each tomo bin

        """

        param_dict = {
            "lens": self.lens_params,
            "source": self.source_params
        }

        params = param_dict[sample_type]
        z_bias = np.repeat(params["z_bias"], params["n_tomo_bins"])
        z_variance = np.repeat(params["sigma_z"], params["n_tomo_bins"])

        return z_bias, z_variance

    def create_redshift_bins(self, redshift_distribution, bin_edges, z_bias_list, z_variance_list):
        """
        Creates the redshift bins for the given distribution and edges.

        Arguments:
            redshift_distribution: array
                The overall redshift distribution.
            bin_edges: array
                The edges of the redshift bins.
            z_bias_list: array
                The list of biases for each bin.
            z_variance_list: array
                The list of variances for each bin.

        Returns:
            redshift_distribution_dict: dict
                A dictionary containing the redshift distributions for each bin.
        """
        redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
            z_bias = float(z_bias_list[index])
            z_variance = float(z_variance_list[index])
            redshift_distribution_dict[index] = self.true_redshift_distribution(redshift_distribution,
                                                                                x1,
                                                                                x2,
                                                                                z_variance,
                                                                                z_bias)
        return redshift_distribution_dict

    def normalize_distribution(self, redshift_distribution, method="trapezoid"):
        """
        Normalizes redshift distributions.

        Arguments:
            redshift_distribution: dict or array
                A dictionary containing redshift distributions for each bin,
                or a single array for an unbinned distribution.
            method: str
                The normalization method to use. Options: 'trapezoid' (default) and 'simpson'.

        Returns:
            normalized_distribution: dict or array
                Normalized distributions, same structure as input.
        """

        def compute_norm_factor(distribution):
            """Helper function to compute normalization factor."""
            if method == "simpson":
                norm_factor = simpson(distribution, x=self.redshift_range)
            elif method == "trapezoid":
                norm_factor = np.trapezoid(distribution, x=self.redshift_range)
            else:
                raise ValueError(f"Unsupported normalization method: {method}. Use 'trapezoid' or 'simpson'.")

            if np.isclose(norm_factor, 0, atol=1e-10):  # Tolerance for handling close to zero values
                raise ValueError("Normalization factor is zero or too small, check the input distribution.")

            return norm_factor

        # Handle single array input
        if isinstance(redshift_distribution, np.ndarray):
            return redshift_distribution / compute_norm_factor(redshift_distribution)

        # Handle dictionary input
        if isinstance(redshift_distribution, dict):
            return {key: dist / compute_norm_factor(dist) for key, dist in redshift_distribution.items()}

        raise TypeError("Input must be a dictionary of distributions or a single numpy array.")

    def get_tomo_bin_edges(self, sample_type):
        """
        Get the redshift tomographic bin edges for the given galaxy sample type.
        Arguments:
            sample_type: str
                the type of the galaxy sample (lens or source)
        Returns:
            bin_edges: array
                the redshift bin edges
        """

        # Get the bin edges as redshift values directly
        sample_dict = {
            "source": self.compute_equal_number_bounds(self.redshift_range,
                                                       self.source_sample(),
                                                       self.source_params["n_tomo_bins"]),
            "lens": np.arange(self.lens_params["bin_start"],
                              self.lens_params["bin_stop"] + self.lens_params["bin_spacing"],
                              self.lens_params["bin_spacing"])
        }

        bin_edges = sample_dict[sample_type]

        return bin_edges

    def compute_tomo_bin_centers(self, bins_dict, decimal_places=2):
        """
        Compute tomographic bin centers from the bins dictionary using numpy.trapezoid.

        Arguments:
            bins_dict (dict): Dictionary of redshift distributions per bin.
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            bin_centers (dict): Dictionary of bin centers with bin indices as keys,
                                rounded to the specified decimal places.
        """
        bin_centers = {}

        for bin_idx, distribution in bins_dict.items():
            weighted_sum = np.trapezoid(distribution * self.redshift_range, self.redshift_range)
            total_area = np.trapezoid(distribution, self.redshift_range)

            # Calculate weighted mean redshift (bin center)
            weighted_mean = weighted_sum / total_area

            # Round the result to the specified number of decimal places
            bin_centers[bin_idx] = round(weighted_mean, decimal_places)

        return bin_centers

    def save_to_file(self, data_name, data, file_format="npy"):
        """
        Save data to a file.

        Arguments:
            data_name: str
                The name of the data.
            data: dict or array
                The data to be saved.
            file_format: str
                The format of the output file (default: 'npy').
                Accepted values are 'csv' and 'npy'.

        Returns:
            None
        """
        # Make sure 'data_output' directory exists
        output_dir = "lsst_nz_and_bins"
        os.makedirs(output_dir, exist_ok=True)

        # Define file path
        file_path = os.path.join(output_dir, f"lsst_{data_name}_year_{self.forecast_year}.{file_format}")

        if file_format == "npy":
            np.save(file_path, data)

        elif file_format == "csv":
            try:
                # Attempt to save directly if data is not nested
                dndz_df = pandas.DataFrame(data)
                dndz_df.to_csv(file_path, index=False)
            except ValueError:
                # Handle nested dictionary cases
                for key, sub_data in data.items():
                    if isinstance(sub_data, dict):
                        sub_df = pandas.DataFrame(sub_data)
                        sub_df.to_csv(f"./lsst_{data_name}_{key}_year_{self.forecast_year}.csv", index=False)

        else:
            raise ValueError(f"Unsupported file format: {file_format}. Use 'npy' or 'csv'.")

    def get_n_eff_clustering(self, decimal_places=2, save_file=False):
        """
        Compute n_eff per clustering bin based on integrated n(z) per bin.
        """
        n_eff_total = self.lens_params["n_gal"]
        z = self.redshift_range
        nz_bin_list = self.lens_bins(normalized=False, save_file=False).values()

        integrals = [np.trapezoid(nz, z) for nz in nz_bin_list]
        total = sum(integrals)
        fractions = [i / total for i in integrals]
        n_eff_list = [round(n_eff_total * f, decimal_places) for f in fractions]

        if save_file:
            self.save_to_file("n_eff_clustering", {"n_gal": n_eff_list}, file_format="npy")

        return n_eff_list

    def get_n_eff_lensing(self, decimal_places=2, save_file=False):
        """
        Split total effective number density equally among source bins.
        """
        n_eff_total = self.source_params["n_gal"]
        num_bins = self.source_params["n_tomo_bins"]
        val = round(n_eff_total / num_bins, decimal_places)
        n_eff_list = [val] * num_bins

        if save_file:
            self.save_to_file("n_eff_lensing", {"n_gal": n_eff_list}, file_format="npy")

        return n_eff_list

    def get_n_eff_frac_clustering(self, decimal_places=4, save_file=False):
        """
        Compute the fraction of n_eff per clustering bin.
        """
        z = self.redshift_range
        nz_bin_list = self.lens_bins(normalized=False, save_file=False).values()

        integrals = [np.trapezoid(nz, z) for nz in nz_bin_list]
        total = sum(integrals)
        fractions = [round(i / total, decimal_places) for i in integrals]

        if save_file:
            self.save_to_file("n_eff_frac_clustering", {"n_eff_frac": fractions}, file_format="npy")

        return fractions

    def get_n_eff_frac_lensing(self, decimal_places=4, save_file=False):
        """
        Compute the fraction of n_eff per lensing (source) bin.
        Since lensing n_eff is split equally, each bin has equal fraction.
        """
        num_bins = self.source_params["n_tomo_bins"]
        frac = round(1.0 / num_bins, decimal_places)
        fractions = [frac] * num_bins

        if save_file:
            self.save_to_file("n_eff_frac_lensing", {"n_eff_frac": fractions}, file_format="npy")

        return fractions
