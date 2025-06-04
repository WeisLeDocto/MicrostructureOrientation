# coding: utf-8

from collections.abc import Sequence
import numpy as np
from scipy.optimize import least_squares, Bounds
from pathlib import Path

from ._kelvin_utils import prepare_data, calc_density, save_results
from ._optimize_diagonals import error_diagonals
from ._optimize_divergence import error_divergence


def _least_square_wrapper(x: np.ndarray,
                          to_fit: np.ndarray,
                          extra_vals: np.ndarray,
                          verbose: bool,
                          include_divergence: bool,
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
        include_divergence: If True, the divergence is computed and used in the
            target function. Otherwise, only the diagonals are use in the
            target function.
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
        lib_path: Path to the .so file containing the shared library for
            computing the stress.
        interp_pts: A numpy array containing all the points over which to
            compute the stress for calculating the final error.
        normals: A numpy array containing for each interpolation point the
            normalized coordinates of its normal along the interpolation line.
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

    if not include_divergence:
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
                               scale,
                               thickness,
                               efforts_x)
    else:
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
                                normals,
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
                       include_divergence: bool,
                       verbose: bool,
                       dest_file: Path,
                       index: int = 0) -> None:
    """Takes a set of images as an input, and finds the best set of material
    parameters that minimizes the error between the measured effort and the
    computed one over all the images on the diagonals.

    Optionally, the divergence can also be computed and included in the target
    function to minimize.

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
        include_divergence: If True, the divergence is computed and used in the
            target function. Otherwise, only the diagonals are use in the
            target function.
        verbose: It True, the values of the parameters are printed at each
            iteration of the optimization loop. Otherwise, they are never
            displayed.
        dest_file: Path to a .csv file where to store the output of the
            optimization.
        index: Index of the processed image in case only one image is being
            processed, otherwise leave to default.
    """

    # Perform a first processing on the input data
    (exxs, eyys, exys,
     sigma_1, sigma_2, sigma_3,
     theta_1, theta_2, theta_3,
     interp_pts, normals) = prepare_data(ref_img,
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
                                'include_divergence': include_divergence,
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
                                'scale': scale,
                                'thickness': thickness,
                                'efforts_x': efforts_x})

    # Save the results to a csv file
    save_results(fit.x,
                 dest_file,
                 order_coeffs,
                 to_fit,
                 extra_vals,
                 index)
