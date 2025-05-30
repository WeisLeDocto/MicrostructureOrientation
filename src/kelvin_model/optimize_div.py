# coding: utf-8

from pathlib import Path
import numpy as np
from collections.abc import Sequence
from scipy.optimize import Bounds, least_squares
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import sobel
from tqdm.auto import tqdm
import sys

from .kelvin_utils import (prepare_data, calc_density, save_results,
                           stress_diag_to_force)
from .compute_stress import compute_stress
from .optimize_diagonals import error_diagonals


def error_div_one_image(lib_path: Path,
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
                        effort_x: float,
                        scale: float,
                        thickness: float,
                        ) -> tuple[float, float]:
    """Computes the stress over the entire image, computes the divergence of
    the stress and estimates the force in the x axis, then returns the norm of
    the divergence and the error in the computed force.

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
        effort_x: The measured effort in x during the test.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.

    Returns:
        The norm of the divergence over the entire image, normalized by various
        factors, and the error in the computed force.
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

    # Calculate the stress values
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

    # Build the divergence tensor
    stress = np.stack((np.stack((sxx, sxy), axis=2),
                       np.stack((sxy, syy), axis=2)), axis=3)
    dxx_dx = sobel(stress[:, :, 0, 0], 1)
    dxy_dx = sobel(stress[:, :, 0, 1], 1)
    dxy_dy = sobel(stress[:, :, 1, 0], 0)
    dyy_dy = sobel(stress[:, :, 1, 1], 0)
    div = np.stack((dxx_dx + dxy_dy, dxy_dx + dyy_dy), axis=2)

    # Compute the norm of the divergence
    error_div = np.sum(np.sqrt(np.sum(np.power(div, 2), axis=-1)), axis=None)
    error_div /= np.sum(np.sqrt(np.sum(np.power(stress, 2), axis=-1)),
                        axis=None)
    error_div /= exx.shape[0] * exx.shape[1]
    # Increase error to match scale of force
    error_div *= 1.0e6

    # Interpolate the stress fields over the provided interpolation points
    sxx_int = RegularGridInterpolator((np.arange(sxx.shape[0]),
                                       np.arange(sxx.shape[1])), sxx)
    syy_int = RegularGridInterpolator((np.arange(syy.shape[0]),
                                       np.arange(syy.shape[1])), syy)
    sxy_int = RegularGridInterpolator((np.arange(sxy.shape[0]),
                                       np.arange(sxy.shape[1])), sxy)
    sxx_diags = sxx_int(interp_pts)
    syy_diags = syy_int(interp_pts)
    sxy_diags = sxy_int(interp_pts)

    # Integrate the stress to get the force over the provided sections
    comp_force_x, _ = stress_diag_to_force(sxx_diags,
                                           syy_diags,
                                           sxy_diags,
                                           density.shape[0],
                                           interp_pts.shape[1],
                                           normals,
                                           cosines,
                                           scale,
                                           thickness)
    error_force = abs(float(np.median(comp_force_x)) - effort_x)

    return error_div, error_force


def error_divergence(lib_path: Path,
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
                     efforts_x: Sequence[float],
                     scale: float,
                     thickness: float,
                     ) -> float:
    """Computes for each image the norm of the divergence of the stress for the
    given set of material parameters, and returns the sum of the errors for all
    images.

    Args:
        lib_path:  Path to the .so file containing the shared library for
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
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.

    Returns:
        The total error as a float, for all the images together.
    """

    # Not really needed as the sections are straight and vertical, but defined
    # for compatibility
    normals = np.zeros_like(interp_pts)
    normals[..., 0] = 1.0
    cosines = np.full_like(interp_pts, 1.0)
    normals = normals[..., np.newaxis]

    nb_tot = len(exxs)

    # Iterate over all the images and get each individual error
    error_div_tot = 0.0
    error_force_tot = 0.0
    for exx, eyy, exy, effort_x in tqdm(zip(exxs, eyys, exys, efforts_x),
                                        total=nb_tot,
                                        desc='Compute the stress for all the '
                                             'images',
                                        file=sys.stdout,
                                        colour='green',
                                        position=1,
                                        leave=False):
        error_div, error_force = error_div_one_image(lib_path,
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
                                                     effort_x,
                                                     scale,
                                                     thickness)
        error_div_tot += error_div
        error_force_tot += error_force

    # Normalize by the total number of images
    error_div_tot /= nb_tot
    error_force_tot /= nb_tot

    # Print the errors if requested
    if verbose:
        print("\n",
              error_div_tot, "\n",
              error_force_tot, "\n",
              error_div_tot + error_force_tot, "\n")

    return error_div_tot + error_force_tot


def _least_square_wrapper(x: np.ndarray,
                          to_fit: np.ndarray,
                          extra_vals: np.ndarray,
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
                          efforts_x: Sequence[float],
                          scale: float,
                          thickness: float,
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
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.

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

    return error_divergence(lib_path,
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
                            efforts_x,
                            scale,
                            thickness)


def _least_square_wrapper_mult(x: np.ndarray,
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
                               min_density: float,
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
                               lib_path: Path,
                               interp_pts: np.ndarray,
                               scale: float,
                               thickness: float,
                               efforts_x: Sequence[float]
                               ) -> float:
    """Parses the arguments from the optimization function, to pass them to the
    function computing the error.

    Args:
        x: Numpy array containing the value to fit.
        verbose: It True, the values of the parameter is printed at each
            iteration of the optimization loop. Otherwise, it is never
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
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        efforts_x: Sequence of floats representing the measured force in the x
            direction, one for each image.

    Returns:
        The total error as a float, for all the images together.
    """

    mult = x[0]
    val1 *= mult
    val2 *= mult
    val3 *= mult
    val4 *= mult
    val5 *= mult

    density = calc_density(density_base, min_density)
    
    if verbose:
        print(mult)

    normals = np.zeros_like(interp_pts)
    normals[..., 0] = 1.0
    cosines = np.full_like(interp_pts, 1.0)

    normals = normals[..., np.newaxis]
    cosines = cosines[..., np.newaxis]

    return error_diagonals(lib_path,
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
                           efforts_x)


def optimize_divergence(lib_path: Path,
                        ref_img: np.ndarray,
                        density_base: np.ndarray,
                        gauss_fit: np.ndarray,
                        peaks: np.ndarray,
                        x0: np.ndarray,
                        def_images: Sequence[np.ndarray],
                        efforts_x: Sequence[float],
                        order_coeffs: np.ndarray,
                        scale: float,
                        thickness: float,
                        verbose: bool,
                        dest_file: Path,
                        cross_section_downscaling: int,
                        index: int = 0) -> None:
    """Takes a set of images as an input, and finds the best set of material
    parameters that minimizes both the norm of the divergence over the entire
    images, and the difference between the computed force and the measured one
    in the x direction.

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
        order_coeffs: Numpy array containing the multiplicative factor for all
            the orders of the material model.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        dest_file: Path to a .csv file where to store the output of the
            optimization.
        cross_section_downscaling: Only one out of this number of sections will
            be used for performing the correction.
        index: Index of the processed image in case only one image is being
            processed, otherwise leave to default.
    """

    # Perform a first processing on the input data
    (exxs, eyys, exys, sigma_1, sigma_2, sigma_3,
     theta_1, theta_2, theta_3, *_) = prepare_data(ref_img,
                                                   gauss_fit,
                                                   peaks,
                                                   def_images,
                                                   None,
                                                   None)

    # Define perpendicular sections
    interp_pts = np.stack(np.meshgrid(np.arange(ref_img.shape[0]),
                                      np.arange(ref_img.shape[1])), axis=2)
    interp_pts = interp_pts[::cross_section_downscaling]

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
    x_scale[5:] = 10.0
    x_scale = x_scale[to_fit]

    # Perform the optimization
    fit = least_squares(_least_square_wrapper,
                        x0,
                        bounds=bounds,
                        x_scale=x_scale,
                        kwargs={'to_fit': to_fit,
                                'extra_vals': extra_vals,
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
                                'efforts_x': efforts_x,
                                'scale': scale,
                                'thickness': thickness})

    # Save the results to a .csv file
    save_results(fit.x,
                 dest_file,
                 order_coeffs,
                 to_fit,
                 extra_vals,
                 index)
