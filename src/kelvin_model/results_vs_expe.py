# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
from collections.abc import Sequence
from tqdm.auto import tqdm
import sys

from .kelvin_utils import (prepare_data, diagonals_interpolator, calc_density,
                           stress_diag_to_force)
from .compute_stress import compute_stress


def compare_results_expe(lib_path: Path,
                         ref_img: np.ndarray,
                         density_base: np.ndarray,
                         gauss_fit: np.ndarray,
                         peaks: np.ndarray,
                         results_file: Path,
                         def_images: Sequence[np.ndarray],
                         timestamps: Sequence[float],
                         efforts_x: Sequence[float],
                         efforts_y: Sequence[float],
                         scale: float,
                         thickness: float,
                         nb_interp_diag: int,
                         diagonal_downscaling: int,
                         dest_file: Path) -> None:
    """

    Args:
        lib_path: Path to the .so file containing the shared library for
          computing the stress.
        ref_img: The reference image for correlation.
        density_base: The base image from which to compute the density,
            normally one the raw images acquired for exposure fusion.
        gauss_fit: Array indicating for each pixel the standard deviation
            obtained by fitting a gaussian curve on the angular response to
            Gabor filter.
        peaks: Array indicating for each pixel the dominant angle for the three
            detected layers of tissue.
        results_file: Path to the file containing the fitted values of the
            material parameters, to use for computing the stress.
        def_images: Sequence of images to use for determining the material
            parameters.
        timestamps: Sequence of floats representing the moment when the images
            were acquired, one for each image.
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.
        efforts_y: Sequence of floats representing the measured force in the y
            direction, one for each image.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        nb_interp_diag: Number of interpolation points along the diagonals.
        diagonal_downscaling: Only one out of this number diagonals will be
            used for performing the optimization.
        dest_file: Path to the file where to write the measured and calculated
            effort values.
    """

    # Perform a first processing on the input data
    (exxs, eyys, exys,
     sigma_1, sigma_2, sigma_3,
     theta_1, theta_2, theta_3,
     interp_pts, normals, cosines) = prepare_data(ref_img,
                                                  gauss_fit,
                                                  peaks,
                                                  def_images,
                                                  nb_interp_diag,
                                                  diagonal_downscaling)

    # Read the fitted material properties from a results file
    result_props = pd.read_csv(results_file)

    # Compute the density from the base image and the minimum density value
    density = calc_density(density_base, result_props["density_min"].iloc[0])

    # Buffers to store the simulated efforts
    forces_x = list()
    forces_y = list()

    # Iterate over all the measurement points for a single test
    for exx, eyy, exy in tqdm(zip(exxs, eyys, exys),
                              total=len(exxs),
                              desc='Predict the effort for all measurement '
                                   'points',
                              file=sys.stdout,
                              colour='green',
                              position=1,
                              leave=False):

        # Interpolate the various fields over the given interpolation points
        (exx_diags, eyy_diags, exy_diags, theta_1_diags, theta_2_diags,
         theta_3_diags, sigma_1_diags, sigma_2_diags, sigma_3_diags,
         density_diags) = diagonals_interpolator(exx,
                                                 eyy,
                                                 exy,
                                                 interp_pts,
                                                 theta_1,
                                                 theta_2,
                                                 theta_3,
                                                 sigma_1,
                                                 sigma_2,
                                                 sigma_3,
                                                 density)

        # Compute the stress fields
        sxx, syy, sxy = compute_stress(lib_path,
                                       exx_diags,
                                       eyy_diags,
                                       exy_diags,
                                       result_props["lambda_h"].iloc[0],
                                       result_props["lambda_11"].iloc[0],
                                       result_props["lambda_21"].iloc[0],
                                       result_props["lambda_51"].iloc[0],
                                       result_props["lambda_12"].iloc[0],
                                       result_props["lambda_22"].iloc[0],
                                       result_props["lambda_52"].iloc[0],
                                       result_props["lambda_13"].iloc[0],
                                       result_props["lambda_23"].iloc[0],
                                       result_props["lambda_53"].iloc[0],
                                       result_props["lambda_14"].iloc[0],
                                       result_props["lambda_24"].iloc[0],
                                       result_props["lambda_54"].iloc[0],
                                       result_props["lambda_15"].iloc[0],
                                       result_props["lambda_25"].iloc[0],
                                       result_props["lambda_55"].iloc[0],
                                       result_props["val1"].iloc[0],
                                       result_props["val2"].iloc[0],
                                       result_props["val3"].iloc[0],
                                       result_props["val4"].iloc[0],
                                       result_props["val5"].iloc[0],
                                       theta_1_diags,
                                       theta_2_diags,
                                       theta_3_diags,
                                       sigma_1_diags,
                                       sigma_2_diags,
                                       sigma_3_diags,
                                       density_diags)

        # Derive the x and y force from the stress fields
        comp_force_x, comp_force_y = stress_diag_to_force(sxx,
                                                          syy,
                                                          sxy,
                                                          density.shape[0],
                                                          interp_pts.shape[1],
                                                          normals,
                                                          cosines,
                                                          scale,
                                                          thickness)

        # Only keep the average force in x and y
        forces_x.append(np.median(comp_force_x))
        forces_y.append(np.median(comp_force_y))

    # Write all the force data into a single file
    results = pd.DataFrame({'time': timestamps,
                            'xx_strain': tuple(np.median(exx)
                                               for exx in exxs),
                            'yy_strain': tuple(np.median(eyy)
                                               for eyy in eyys),
                            'xy_strain': tuple(np.median(exy)
                                               for exy in exys),
                            'measured_x': efforts_x,
                            'calculated_x': forces_x,
                            'measured_y': efforts_y,
                            'calculated_y': forces_y})
    results.to_csv(dest_file, index=False)
