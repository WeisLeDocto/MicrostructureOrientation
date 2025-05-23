# coding: utf-8

from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, Bounds
from pathlib import Path
from tqdm.auto import tqdm
import sys
import concurrent.futures
import itertools
from multiprocessing.synchronize import RLock as RLockType
from multiprocessing import RLock

from .compute_stress import compute_stress
from .kelvin_utils import (prepare_data, diagonals_interpolator, calc_density,
                           stress_diag_to_force)


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
    return abs(float(np.median(comp_force_x)) - effort_x)


def _process_pool_wrapper(args: tuple[Path, np.ndarray, np.ndarray, np.ndarray, 
                                      float, float, float, float, float, float, 
                                      float, float, float, float, float, float, 
                                      float, float, float, float, float, float, 
                                      float, float, float, np.ndarray, 
                                      np.ndarray, np.ndarray, np.ndarray, 
                                      np.ndarray, np.ndarray, np.ndarray, 
                                      np.ndarray, np.ndarray, np.ndarray, 
                                      float, float, float]) -> float:
    """Wrapper for passing the arguments separately to the worker, for better
    clarity.

    Args:
        args: The arguments to pass to the worker, as a single tuple.

    Returns:
        The computed error as a float.
    """

    return error_diags_one_image(*args)


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
    the error for all images.

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
    if nb_tot == 1:
        return error_diags_one_image(lib_path,
                                     exxs[0],
                                     eyys[0],
                                     exys[0],
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
                                     efforts_x[0])
    
    # Arrange the arguments to pass the to the ProcessPoolExecutor
    args = zip(itertools.repeat(lib_path, nb_tot),
               exxs,
               eyys,
               exys,
               itertools.repeat(lambda_h, nb_tot),
               itertools.repeat(lambda_11, nb_tot),
               itertools.repeat(lambda_21, nb_tot),
               itertools.repeat(lambda_51, nb_tot),
               itertools.repeat(lambda_12, nb_tot),
               itertools.repeat(lambda_22, nb_tot),
               itertools.repeat(lambda_52, nb_tot),
               itertools.repeat(lambda_13, nb_tot),
               itertools.repeat(lambda_23, nb_tot),
               itertools.repeat(lambda_53, nb_tot),
               itertools.repeat(lambda_14, nb_tot),
               itertools.repeat(lambda_24, nb_tot),
               itertools.repeat(lambda_54, nb_tot),
               itertools.repeat(lambda_15, nb_tot),
               itertools.repeat(lambda_25, nb_tot),
               itertools.repeat(lambda_55, nb_tot),
               itertools.repeat(val1, nb_tot),
               itertools.repeat(val2, nb_tot),
               itertools.repeat(val3, nb_tot),
               itertools.repeat(val4, nb_tot),
               itertools.repeat(val5, nb_tot),
               itertools.repeat(theta_1, nb_tot),
               itertools.repeat(theta_2, nb_tot),
               itertools.repeat(theta_3, nb_tot),
               itertools.repeat(sigma_1, nb_tot),
               itertools.repeat(sigma_2, nb_tot),
               itertools.repeat(sigma_3, nb_tot),
               itertools.repeat(density, nb_tot),
               itertools.repeat(interp_pts, nb_tot),
               itertools.repeat(normals, nb_tot),
               itertools.repeat(cosines, nb_tot),
               itertools.repeat(scale, nb_tot),
               itertools.repeat(thickness, nb_tot),
               efforts_x)
    
    # Distribute the processing of the error on all the cores
    error_tot = 0.0
    with tqdm(total=nb_tot,
              desc='Compute the stress for all the images',
              file=sys.stdout,
              colour='green',
              position=1,
              leave=False) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for error in executor.map(_process_pool_wrapper, 
                                      args, chunksize=1):
                error_tot += error
                pbar.update()
    
    # Print the total error if requested
    if verbose:
        print("\n", error_tot, "\n")
    return error_tot


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
                          normals: np.ndarray,
                          cosines: np.ndarray,
                          scale: float,
                          thickness: float,
                          efforts_x: Sequence[float]
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


def optimize_diagonals(lib_path: Path,
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
                       nb_interp_diag: int,
                       diagonal_downscaling: int,
                       verbose: bool,
                       dest_file: Path,
                       index: int = 0,
                       lock: RLockType | None = None) -> None:
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
        order_coeffs: Numpy array containing the multiplicative factor for all
            the orders of the material model.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.
        nb_interp_diag: Number of interpolation points along the diagonals.
        diagonal_downscaling: Only one out of this number diagonals will be
            used for performing the optimization.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        dest_file: Path to a .npy file where to store the output of the
            optimization.
        index: Index of the processed image in case only one image is being
            processed, otherwise leave to default.
        lock: A lock to avoid concurrency between processes when writing in the
           results file.
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
                                'normals': normals,
                                'cosines': cosines,
                                'scale': scale,
                                'thickness': thickness,
                                'efforts_x': efforts_x})

    # The labels of all the value to save for making the model
    labels = ("idx", "val1", "val2", "val3", "val4", "val5", "density_min",
              "lambda_h", "lambda_11", "lambda_21", "lambda_51", "lambda_12",
              "lambda_22", "lambda_52", "lambda_13", "lambda_23", "lambda_53",
              "lambda_14", "lambda_24", "lambda_54", "lambda_15", "lambda_25",
              "lambda_55")

    # If no lock was given just create a useless one
    if lock is None:
        lock = RLock()

    # Protect writing to results file with lock
    with lock:

        # Load previous results if they already exist at the indicated location
        if dest_file.exists() and dest_file.is_file():
            results = pd.read_csv(dest_file)
        else:
            results = pd.DataFrame(columns=labels)

        # Store the various order coefficients
        new_vals = pd.Series()
        new_vals[labels[0]] = index

        for label, val in zip(labels[1:6], order_coeffs.tolist(), strict=True):
            new_vals[label] = val

        # Store all the other values for the material coefficients
        extra = iter(extra_vals.tolist())
        fitted = iter(fit.x.tolist())
        for label, flag in zip(labels[6:], to_fit.tolist(), strict=True):
            if flag:
                new_vals[label] = next(fitted)
            else:
                new_vals[label] = next(extra)

        # Fuse the new results with the existing ones and save to a csv file
        results = pd.concat((results, new_vals.to_frame().transpose()),
                            ignore_index=True)
        results.to_csv(dest_file, columns=labels, index=False)
