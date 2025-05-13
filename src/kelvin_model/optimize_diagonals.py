# coding: utf-8

from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, Bounds
from pathlib import Path
from tqdm.auto import tqdm
import sys

from .compute_stress import compute_stress
from .kelvin_utils import (prepare_data, diagonals_interpolator, calc_density,
                           stress_diag_to_force)


def error_all_image(lib_path: Path,
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
                    effort_x: float,
                    effort_y: float) -> float:
    """Computes the stress on the entire image at once, then extracts it on the
    diagonals of interest, and returns the error in the computed x and y force.

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
        effort_y: The measured effort in y during the test.

    Returns:
        The error in the computed force along the diagonals, normalized by
        various factors.
    """

    # Compute the stress on the entire image at once
    sxx, syy, sxy = compute_stress(lib_path,
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
                                   density)

    # Build interpolators for the stress fields and compute the stress on the
    # provided diagonals
    sxx_int = RegularGridInterpolator((np.arange(sxx.shape[0]),
                                       np.arange(sxx.shape[1])), sxx)
    syy_int = RegularGridInterpolator((np.arange(syy.shape[0]),
                                       np.arange(syy.shape[1])), syy)
    sxy_int = RegularGridInterpolator((np.arange(sxy.shape[0]),
                                       np.arange(sxy.shape[1])), sxy)
    sxx_diags = sxx_int(interp_pts)
    syy_diags = syy_int(interp_pts)
    sxy_diags = sxy_int(interp_pts)

    # Normalize by the number of diagonals because the more diagonals the
    # greater the error
    downscale_factor_w = interp_pts.shape[0]
    # Normalize by the effort to get a relative error
    effort_norm = np.sqrt(effort_x ** 2 + effort_y ** 2)

    comp_force_x, comp_force_y = stress_diag_to_force(sxx_diags,
                                                      syy_diags,
                                                      sxy_diags,
                                                      density.shape[0],
                                                      interp_pts.shape[1],
                                                      normals,
                                                      cosines,
                                                      scale,
                                                      thickness)

    # Return the error as the difference between the computed and expect effort
    return np.sum(np.sqrt(np.power(comp_force_x - effort_x, 2) +
                          np.power(comp_force_y - effort_y, 2)),
                  axis=None) / downscale_factor_w / effort_norm


def error_diags_only(lib_path: Path,
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
                     effort_x: float,
                     effort_y: float) -> float:
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
        effort_y: The measured effort in y during the test.

    Returns:
        The error in the computed force along the diagonals, normalized by
        various factors.
    """

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

    # Compute the stress only on the diagonals
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

    # Normalize by the number of diagonals because the more diagonals the
    # greater the error
    downscale_factor_w = interp_pts.shape[0]
    # Normalize by the effort to get a relative error
    effort_norm = np.sqrt(effort_x ** 2 + effort_y ** 2)

    comp_force_x, comp_force_y = stress_diag_to_force(sxx,
                                                      syy,
                                                      sxy,
                                                      density.shape[0],
                                                      interp_pts.shape[1],
                                                      normals,
                                                      cosines,
                                                      scale,
                                                      thickness)

    # Return the error as the difference between the computed and expect effort
    return np.sum(np.sqrt(np.power(comp_force_x - effort_x, 2) +
                          np.power(comp_force_y - effort_y, 2)),
                  axis=None) / downscale_factor_w / effort_norm


def error_diagonals(lib_path: Path,
                    interp_strain: bool,
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
                    efforts_x: Sequence[float],
                    efforts_y: Sequence[float]) -> float:
    """Computes for each image the error in the calculated forces along
    diagonals for the given set of material parameters, and returns the sum of
    the error for all images.

    Args:
        lib_path:  Path to the .so file containing the shared library for
          computing the stress.
        interp_strain: If True, performs diagonal interpolation on the strain
            before computing the stress, otherwise performs interpolation on
            the stress after computing it for the entire image.
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
        efforts_y: Sequence of floats representing the measured force in the y
            direction, one for each image.

    Returns:
        The total error as a float, for all the images together.
    """

    # Select the appropriate error function
    err_func = error_diags_only if interp_strain else error_all_image

    # Compute the error for all the provided images, and store one global
    # error value
    error = 0.0
    for (exx, eyy, exy,
         effort_x, effort_y) in tqdm(zip(exxs,
                                         eyys,
                                         exys,
                                         efforts_x,
                                         efforts_y),
                                     total=len(exxs),
                                     desc='Compute the stress for all images',
                                     file=sys.stdout,
                                     colour='green',
                                     position=1,
                                     leave=False):
        error += err_func(lib_path,
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
                          effort_x,
                          effort_y)

    if verbose:
        print("\n", error, "\n")
    return error


def _wrapper(x: np.ndarray,
             to_fit: np.ndarray,
             extra_vals: np.ndarray,
             interp_strain: bool,
             verbose: bool,
             val1: float,
             val2: float,
             val3: float,
             val4: float,
             val5: float,
             exxs: Sequence[np.ndarray],
             eyys: Sequence[np.ndarray],
             exys: Sequence[np.ndarray],
             theta_1: np.ndarray,
             theta_2: np.ndarray,
             theta_3: np.ndarray,
             sigma_1: np.ndarray,
             sigma_2: np.ndarray,
             sigma_3: np.ndarray,
             density_base: np.ndarray,
             lib_path: Path,
             interp_pts: np.ndarray,
             normals: np.ndarray,
             cosines: np.ndarray,
             scale: float,
             thickness: float,
             efforts_x: Sequence[float],
             efforts_y: Sequence[float]
             ) -> float:
    """Parses the arguments from the optimization function, to pass them to the
    function computing the error.

    Args:
        x: Numpy array containing the values to fit, which are a subset of all
            the material parameters.
        to_fit: Mask indicating for each material parameter whether it is to
            fit or if it is fixed.
        extra_vals: Numpy array containing the fixed values, which are a subset
            of all the material parameters.
        interp_strain: If True, performs diagonal interpolation on the strain
            before computing the stress, otherwise performs interpolation on
            the stress after computing it for the entire image.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        val1: Multiplicative factor for the first-order term.
        val2: Multiplicative factor for the second-order term.
        val3: Multiplicative factor for the third-order term.
        val4: Multiplicative factor for the fourth-order term.
        val5: Multiplicative factor for the fifth-order term.
        exxs: Sequence of numpy array containing for all pixels the xx strain,
            one for each image.
        eyys: Sequence of numpy array containing for all pixels the yy strain,
            one for each image.
        exys: Sequence of numpy array containing for all pixels the xy strain,
            one for each image.
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
        density_base: The base image from which to compute the density,
            normally one the raw images acquired for exposure fusion.
        lib_path:  Path to the .so file containing the shared library for
          computing the stress.
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
        efforts_y: Sequence of floats representing the measured force in the y
            direction, one for each image.

    Returns:
        The total error as a float, for all the images together.
    """

    # Initialize the objects containing the material parameters
    lambdas = np.empty(16, dtype=np.float64)
    x = iter(x)
    extra_vals = iter(extra_vals)

    # The first value is always the minimum material density
    dens_min = next(x)

    # Retrieve all the material parameters from the correct iterables
    for i, flag in enumerate(to_fit[1:].tolist()):
        if flag:
            lambdas[i] = next(x)
        else:
            lambdas[i] = next(extra_vals)

    (lambda_h,
     lambda_11, lambda_21, lambda_51,
     lambda_12, lambda_22, lambda_52,
     lambda_13, lambda_23, lambda_53,
     lambda_14, lambda_24, lambda_54,
     lambda_15, lambda_25, lambda_55) = lambdas

    # If requested, print the values of the material parameters
    if verbose:
        print(dens_min, lambda_h,
              lambda_11, lambda_21, lambda_51,
              lambda_12, lambda_22, lambda_52,
              lambda_13, lambda_23, lambda_53,
              lambda_14, lambda_24, lambda_54,
              lambda_15, lambda_25, lambda_55)

    # Compute the density from the base image and the minimum density value
    density = calc_density(density_base, dens_min)

    return error_diagonals(lib_path,
                           interp_strain,
                           verbose,
                           exxs,
                           eyys,
                           exys,
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
                           efforts_x,
                           efforts_y)


def optimize_diagonals(lib_path: Path,
                       ref_img: np.ndarray,
                       density_base: np.ndarray,
                       gauss_fit: np.ndarray,
                       peaks: np.ndarray,
                       x0: np.ndarray,
                       def_images: Sequence[np.ndarray],
                       efforts_x: Sequence[float],
                       efforts_y: Sequence[float],
                       order_coeffs: np.ndarray,
                       scale: float,
                       thickness: float,
                       nb_interp_diag: int,
                       interp_strain: bool,
                       diagonal_downscaling: int,
                       verbose: bool,
                       dest_file: Path) -> None:
    """Takes a set of images as an input, and finds the best set of material
    parameters that minimizes the error between the measured effort and the
    computed one over all the images on the diagonals.

    Args:
        lib_path:  Path to the .so file containing the shared library for
          computing the stress.
        ref_img: The reference image for correlation.
        density_base: The base image from which to compute the density,
            normally one the raw images acquired for exposure fusion.
        gauss_fit: Array indicating for each pixel the standard deviation
            obtained by fitting a gaussian curve on the angular response to
            Gabor filter.
        peaks: Array indicating for each pixel the dominant angle for the three
            detected layers of tissue.
        x0: Initial gess for the optimized parameters, as a numpy array.
        def_images: Sequence of images to use for determining the material
            parameters.
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.
        efforts_y: Sequence of floats representing the measured force in the y
            direction, one for each image.
        order_coeffs: Numpy array containing the multiplicative factor for all
            the orders of the material model.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        nb_interp_diag: Number of interpolation points along the diagonals.
        interp_strain: If True, performs diagonal interpolation on the strain
            before computing the stress, otherwise performs interpolation on
            the stress after computing it for the entire image.
        diagonal_downscaling: Only one out of this number diagonals will be
            used for performing the optimization.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        dest_file: Path to a .npy file where to store the output of the
            optimization.
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

    # Define the bounds for all the parameters
    nb_max = 17
    low_bounds = np.array((1.0e-5,) * nb_max)
    low_bounds[0] = 0.0
    low_bounds[1] = 1.0
    high_bounds = np.array((np.inf,) * nb_max)
    high_bounds[0] = 0.99

    # Create the lists of parameters to fit or fixed, depending on the values
    # of the factors for each order
    to_fit = np.full(nb_max, True, dtype=np.bool_)
    for idx in np.where(order_coeffs == 0.0)[0].tolist():
        to_fit[2 + 3 * idx: 2 + 3 * (idx + 1)] = False
    low_bounds = low_bounds[to_fit]
    high_bounds = high_bounds[to_fit]
    extra_vals = x0[~to_fit]
    x0 = x0[to_fit]
    bounds = Bounds(lb=low_bounds, ub=high_bounds)
    x_scale = np.ones(nb_max, dtype=np.float64)
    x_scale = x_scale[to_fit]

    # Perform the optimization
    fit = least_squares(_wrapper, x0, bounds=bounds, x_scale=x_scale,
                        kwargs={'to_fit': to_fit,
                                'extra_vals': extra_vals,
                                'interp_strain': interp_strain,
                                'verbose': verbose,
                                'val1': order_coeffs[0],
                                'val2': order_coeffs[1],
                                'val3': order_coeffs[2],
                                'val4': order_coeffs[3],
                                'val5': order_coeffs[4],
                                'exxs': exxs,
                                'eyys': eyys,
                                'exys': exys,
                                'theta_1': theta_1,
                                'theta_2': theta_2,
                                'theta_3': theta_3,
                                'sigma_1': sigma_1,
                                'sigma_2': sigma_2,
                                'sigma_3': sigma_3,
                                'density_base': density_base,
                                'lib_path': lib_path,
                                'interp_pts': interp_pts,
                                'normals': normals,
                                'cosines': cosines,
                                'scale': scale,
                                'thickness': thickness,
                                'efforts_x': efforts_x,
                                'efforts_y': efforts_y})

    # The labels of all the value to save for making the model
    labels = ("val1", "val2", "val3", "val4", "val5", "density_min",
              "lambda_h", "lambda_11", "lambda_21", "lambda_51", "lambda_12",
              "lambda_22", "lambda_52", "lambda_13", "lambda_23", "lambda_53",
              "lambda_14", "lambda_24", "lambda_54", "lambda_15", "lambda_25",
              "lambda_55")

    # Load previous results if they already exist at the indicated location
    if dest_file.exists() and dest_file.is_file():
        results = pd.read_csv(dest_file)
    else:
        results = pd.DataFrame(columns=labels)

    # Store the various order coefficients
    new_vals = pd.Series()
    for label, val in zip(labels[:5], order_coeffs.tolist(), strict=True):
        new_vals[label] = val

    # Store all the other values for the material coefficients
    extra = iter(extra_vals.tolist())
    fitted = iter(fit.x.tolist())
    for label, flag in zip(labels[5:], to_fit.tolist(), strict=True):
        if flag:
            new_vals[label] = next(fitted)
        else:
            new_vals[label] = next(extra)

    # Fuse the new results with the existing ones and save to a csv file
    print(new_vals)
    results = pd.concat((results, new_vals.to_frame().transpose()),
                        ignore_index=True)
    print(results)
    results.to_csv(dest_file, columns=labels, index=False)
