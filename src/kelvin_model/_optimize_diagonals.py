# coding: utf-8

from collections.abc import Sequence
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import sys

from .compute_stress import compute_stress
from ._kelvin_utils import diagonals_interpolator, stress_diag_to_force


def error_diags_one_image(lib_path: Path,
                          exx: np.ndarray,
                          eyy: np.ndarray,
                          exy: np.ndarray,
                          lambda_h: float,
                          lambda_11: float,
                          lambda_21: float,
                          lambda_51: float,
                          lambda_12: float,
                          lambda_22: float,
                          lambda_52: float,
                          lambda_13: float,
                          lambda_23: float,
                          lambda_53: float,
                          lambda_14: float,
                          lambda_24: float,
                          lambda_54: float,
                          lambda_15: float,
                          lambda_25: float,
                          lambda_55: float,
                          val1: float,
                          val2: float,
                          val3: float,
                          val4: float,
                          val5: float,
                          theta_1: np.ndarray,
                          theta_2: np.ndarray,
                          theta_3: np.ndarray,
                          sigma_1: np.ndarray,
                          sigma_2: np.ndarray,
                          sigma_3: np.ndarray,
                          density: np.ndarray,
                          interp_pts: np.ndarray,
                          normals: np.ndarray,
                          cosines: np.ndarray,
                          scale: float,
                          thickness: float,
                          effort_x: float) -> float:
    """Computes the stress only on the provided diagonals, and returns the
    error in the computed x and y force.

    Args:
        lib_path: Path to the .so file containing the shared library for
            computing the stress.
        exx: Numpy array containing for all pixels the xx strain.
        eyy: Numpy array containing for all pixels the yy strain.
        exy: Numpy array containing for all pixels the xy strain.
        lambda_h: Value of the hydrostatic eigenvalue, common to all orders.
        lambda_11: Value of the first deviatoric eigenvalue for order 1.
        lambda_21: Value of the second deviatoric eigenvalue for order 1.
        lambda_51: Value of the fifth deviatoric eigenvalue for order 1.
        lambda_12: Value of the first deviatoric eigenvalue for order 2.
        lambda_22: Value of the second deviatoric eigenvalue for order 2.
        lambda_52: Value of the fifth deviatoric eigenvalue for order 2.
        lambda_13: Value of the first deviatoric eigenvalue for order 3.
        lambda_23: Value of the second deviatoric eigenvalue for order 3.
        lambda_53: Value of the fifth deviatoric eigenvalue for order 3.
        lambda_14: Value of the first deviatoric eigenvalue for order 4.
        lambda_24: Value of the second deviatoric eigenvalue for order 4.
        lambda_54: Value of the fifth deviatoric eigenvalue for order 4.
        lambda_15: Value of the first deviatoric eigenvalue for order 5.
        lambda_25: Value of the second deviatoric eigenvalue for order 5.
        lambda_55: Value of the fifth deviatoric eigenvalue for order 5.
        val1: Multiplicative factor for the first-order term.
        val2: Multiplicative factor for the second-order term.
        val3: Multiplicative factor for the third-order term.
        val4: Multiplicative factor for the fourth-order term.
        val5: Multiplicative factor for the fifth-order term.
        theta_1: Numpy array containing for all pixels the local angle of the
            first tissue layer.
        theta_2: Numpy array containing for all pixels the local angle of the
            second tissue layer.
        theta_3: Numpy array containing for all pixels the local angle of the
            third tissue layer.
        sigma_1: Numpy array containing for all pixels the local standard
            deviation of the first tissue layer.
        sigma_2: Numpy array containing for all pixels the local standard
            deviation of the second tissue layer.
        sigma_3: Numpy array containing for all pixels the local standard
            deviation of the third tissue layer.
        density: Numpy array containing for all pixels the local density of the
            tissue.
        interp_pts: A numpy array containing all the points over which to
            compute the stress for calculating the final error.
        normals: A numpy array containing for each interpolation point the
            normalized coordinates of its normal along the interpolation line.
        cosines: A numpy array containing for each interpolation point the
            scaling factor to use for correcting the inclination of the
            interpolation line.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        effort_x: The measured effort in x during the test.

    Returns:
        The error in the computed force along the diagonals, normalized by
        various factors.
    """
    
    # Ensure there is no nan value in the data
    exx = np.nan_to_num(exx)
    eyy = np.nan_to_num(eyy)
    exy = np.nan_to_num(exy)
    theta_1 = np.nan_to_num(theta_1)
    theta_2 = np.nan_to_num(theta_2)
    theta_3 = np.nan_to_num(theta_3)
    sigma_1 = np.nan_to_num(sigma_1)
    sigma_2 = np.nan_to_num(sigma_2)
    sigma_3 = np.nan_to_num(sigma_3)
    density = np.nan_to_num(density)
    
    # Interpolate the fields on the diagonals
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
    
    # Calculate the stress values
    sxx, syy, sxy = compute_stress(lib_path,
                                   exx_diags,
                                   eyy_diags,
                                   exy_diags,
                                   lambda_h,
                                   lambda_11,
                                   lambda_21,
                                   lambda_51,
                                   lambda_12,
                                   lambda_22,
                                   lambda_52,
                                   lambda_13,
                                   lambda_23,
                                   lambda_53,
                                   lambda_14,
                                   lambda_24,
                                   lambda_54,
                                   lambda_15,
                                   lambda_25,
                                   lambda_55,
                                   val1,
                                   val2,
                                   val3,
                                   val4,
                                   val5,
                                   theta_1_diags,
                                   theta_2_diags,
                                   theta_3_diags,
                                   sigma_1_diags,
                                   sigma_2_diags,
                                   sigma_3_diags,
                                   density_diags)
        
    # Derive the force from the stress fields
    comp_force_x, _ = stress_diag_to_force(sxx,
                                           syy,
                                           sxy,
                                           density.shape[0],
                                           interp_pts.shape[1],
                                           normals,
                                           cosines,
                                           scale,
                                           thickness)
    
    # Return the error as the difference between the computed and expect effort
    return abs(float(np.median(comp_force_x)) - effort_x) / effort_x


def error_diagonals(lib_path: Path,
                    verbose: bool,
                    exxs: Sequence[np.ndarray],
                    eyys: Sequence[np.ndarray],
                    exys: Sequence[np.ndarray],
                    lambda_h: float,
                    lambda_11: float,
                    lambda_21: float,
                    lambda_51: float,
                    lambda_12: float,
                    lambda_22: float,
                    lambda_52: float,
                    lambda_13: float,
                    lambda_23: float,
                    lambda_53: float,
                    lambda_14: float,
                    lambda_24: float,
                    lambda_54: float,
                    lambda_15: float,
                    lambda_25: float,
                    lambda_55: float,
                    val1: float,
                    val2: float,
                    val3: float,
                    val4: float,
                    val5: float,
                    theta_1: np.ndarray,
                    theta_2: np.ndarray,
                    theta_3: np.ndarray,
                    sigma_1: np.ndarray,
                    sigma_2: np.ndarray,
                    sigma_3: np.ndarray,
                    density: np.ndarray,
                    interp_pts: np.ndarray,
                    normals: np.ndarray,
                    cosines: np.ndarray,
                    scale: float,
                    thickness: float,
                    efforts_x: Sequence[float]) -> float:
    """Computes for each image the error in the calculated forces along
    diagonals for the given set of material parameters, and returns the sum of
    the errors for all images.

    Args:
        lib_path: Path to the .so file containing the shared library for
            computing the stress.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        exxs: Sequence of numpy array containing for all pixels the xx strain,
            one for each image.
        eyys: Sequence of numpy array containing for all pixels the yy strain,
            one for each image.
        exys: Sequence of numpy array containing for all pixels the xy strain,
            one for each image.
        lambda_h: Value of the hydrostatic eigenvalue, common to all orders.
        lambda_11: Value of the first deviatoric eigenvalue for order 1.
        lambda_21: Value of the second deviatoric eigenvalue for order 1.
        lambda_51: Value of the fifth deviatoric eigenvalue for order 1.
        lambda_12: Value of the first deviatoric eigenvalue for order 2.
        lambda_22: Value of the second deviatoric eigenvalue for order 2.
        lambda_52: Value of the fifth deviatoric eigenvalue for order 2.
        lambda_13: Value of the first deviatoric eigenvalue for order 3.
        lambda_23: Value of the second deviatoric eigenvalue for order 3.
        lambda_53: Value of the fifth deviatoric eigenvalue for order 3.
        lambda_14: Value of the first deviatoric eigenvalue for order 4.
        lambda_24: Value of the second deviatoric eigenvalue for order 4.
        lambda_54: Value of the fifth deviatoric eigenvalue for order 4.
        lambda_15: Value of the first deviatoric eigenvalue for order 5.
        lambda_25: Value of the second deviatoric eigenvalue for order 5.
        lambda_55: Value of the fifth deviatoric eigenvalue for order 5.
        val1: Multiplicative factor for the first-order term.
        val2: Multiplicative factor for the second-order term.
        val3: Multiplicative factor for the third-order term.
        val4: Multiplicative factor for the fourth-order term.
        val5: Multiplicative factor for the fifth-order term.
        theta_1: Numpy array containing for all pixels the local angle of the
            first tissue layer.
        theta_2: Numpy array containing for all pixels the local angle of the
            second tissue layer.
        theta_3: Numpy array containing for all pixels the local angle of the
            third tissue layer.
        sigma_1: Numpy array containing for all pixels the local standard
            deviation of the first tissue layer.
        sigma_2: Numpy array containing for all pixels the local standard
            deviation of the second tissue layer.
        sigma_3: Numpy array containing for all pixels the local standard
            deviation of the third tissue layer.
        density: Numpy array containing for all pixels the local density of the
            tissue.
        interp_pts: A numpy array containing all the points over which to
            compute the stress for calculating the final error.
        normals: A numpy array containing for each interpolation point the
            normalized coordinates of its normal along the interpolation line.
        cosines: A numpy array containing for each interpolation point the
            scaling factor to use for correcting the inclination of the
            interpolation line.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.

    Returns:
        The total error as a float, for all the images together.
    """

    nb_tot = len(exxs)

    # Iterate over all the images and get each individual error
    error_tot = 0.0
    for exx, eyy, exy, effort_x in tqdm(zip(exxs, eyys, exys, efforts_x),
                                        total=nb_tot,
                                        desc='Compute the stress for all the '
                                             'images',
                                        file=sys.stdout,
                                        colour='green',
                                        position=1,
                                        leave=False):
        error_tot += error_diags_one_image(lib_path,
                                           exx,
                                           eyy,
                                           exy,
                                           lambda_h,
                                           lambda_11,
                                           lambda_21,
                                           lambda_51,
                                           lambda_12,
                                           lambda_22,
                                           lambda_52,
                                           lambda_13,
                                           lambda_23,
                                           lambda_53,
                                           lambda_14,
                                           lambda_24,
                                           lambda_54,
                                           lambda_15,
                                           lambda_25,
                                           lambda_55,
                                           val1,
                                           val2,
                                           val3,
                                           val4,
                                           val5,
                                           theta_1,
                                           theta_2,
                                           theta_3,
                                           sigma_1,
                                           sigma_2,
                                           sigma_3,
                                           density,
                                           interp_pts,
                                           normals,
                                           cosines,
                                           scale,
                                           thickness,
                                           effort_x)
    error_tot /= nb_tot
    
    # Print the total error if requested
    if verbose:
        print("\n", error_tot, "\n")
    return error_tot
