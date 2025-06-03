# coding: utf-8

import numpy as np
from collections.abc import Sequence
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from pathlib import Path

from .image_correlation import image_correlation


def prepare_data(ref_img: np.ndarray,
                 gauss_fit: np.ndarray,
                 peaks: np.ndarray,
                 def_images: Sequence[np.ndarray],
                 nb_interp_diag: int,
                 diagonal_downscaling: int
                 ) -> tuple[Sequence[np.ndarray],
                            Sequence[np.ndarray],
                            Sequence[np.ndarray],
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray]:
    """Takes the input data and computes various other values from it.

    Convenience function that avoids repeating the same operations at multiple
    places.

    Args:
        ref_img: The reference image for correlation.
        gauss_fit: Array indicating for each pixel the standard deviation
            obtained by fitting a gaussian curve on the angular response to
            Gabor filter.
        peaks: Array indicating for each pixel the dominant angle for the three
            detected layers of tissue.
        def_images: Sequence of images to use for determining the material
            parameters.
        nb_interp_diag: Number of interpolation points along the diagonals.
        diagonal_downscaling: Only one out of this number diagonals will be
            used for performing the optimization.

    Returns:
        In this order: a sequence of exx strain arrays, a sequence of eyy
        strain array, a sequence of exy strain arrays, the standard deviation
        array for the first tissue layer, the standard deviation array for the
        second tissue layer, the standard deviation array for the third tissue
        layer, the local angle array for the first tissue layer, the local
        angle array for the second tissue layer, the local angle array for the
        third tissue layer, an array containing the coordinates of the diagonal
        points, an array containing the coordinates of the normals to the
        diagonal points, and an array containing the correction factor for the
        inclination of the diagonals in each interpolation point.
    """

    # First, compute the strain using digital image correlation
    exxs, eyys, exys = zip(*(image_correlation(ref_img, def_img)
                             for def_img in def_images))

    # Read the angle and standard deviation values from the input data
    sigma_1 = gauss_fit[:, :, 0]
    sigma_2 = np.nan_to_num(gauss_fit[:, :, 2])
    sigma_3 = np.nan_to_num(gauss_fit[:, :, 4])
    theta_1 = peaks[:, :, 0]
    theta_2 = np.nan_to_num(peaks[:, :, 1])
    theta_3 = np.nan_to_num(peaks[:, :, 2])

    # Generate the interpolation points on the diagonals
    interp_pts = np.empty((ref_img.shape[1], nb_interp_diag, 2),
                          dtype=np.float64)
    for j in range(interp_pts.shape[0]):
        interp_pts[j, :, 1] = np.linspace(j, ref_img.shape[1] - 1 - j,
                                          nb_interp_diag)
        interp_pts[j, :, 0] = np.linspace(0, ref_img.shape[0] - 1,
                                          nb_interp_diag)
    interp_pts = interp_pts[::diagonal_downscaling]

    # Generate the normals to the diagonals on each interpolation point
    normals = np.zeros((ref_img.shape[1], nb_interp_diag, 2),
                       dtype=np.float64)
    for i in range(normals.shape[0]):
        normals[i] = np.array((1.0, (ref_img.shape[1] - 2 * i - 1) /
                               ref_img.shape[0]),
                              dtype=np.float64)
        normals[i] /= np.linalg.norm(normals[i], axis=1)[:, np.newaxis]
    normals = normals[::diagonal_downscaling]

    # The correction factor for the inclination of the diagonals
    cosines = normals @ np.array((1.0, 0.0), dtype=np.float64)

    # Necessary for matrix multiplication later on
    normals = normals[..., np.newaxis]
    cosines = cosines[..., np.newaxis]

    return (exxs, eyys, exys, sigma_1, sigma_2, sigma_3, theta_1, theta_2,
            theta_3, interp_pts, normals, cosines)


def diagonals_interpolator(exx: np.ndarray,
                           eyy: np.ndarray,
                           exy: np.ndarray,
                           interp_pts: np.ndarray,
                           theta_1: np.ndarray,
                           theta_2: np.ndarray,
                           theta_3: np.ndarray,
                           sigma_1: np.ndarray,
                           sigma_2: np.ndarray,
                           sigma_3: np.ndarray,
                           density: np.ndarray
                           ) -> tuple[np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray]:
    """Performs interpolation of the various fields describing the samples on
    the provided interpolation points, and returns the corresponding
    interpolated fields as numpy arrays.

    Args:
        exx: Numpy array containing for all pixels the xx strain.
        eyy: Numpy array containing for all pixels the yy strain.
        exy: Numpy array containing for all pixels the xy strain.
        interp_pts: A numpy array containing all the points over which to
            compute the stress for calculating the final error.
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

    Returns:
        The exx strain, eyy strain, exy strain, angle for the first layer,
        angle for the second layer, angle for the third layer, standard
        deviation for the first layer, standard deviation for the second layer,
        standard deviation for the third layer, and density, interpolated over
        the provided interpolation points.
    """

    # Build interpolators for the strain fields and compute the strains on the
    # provided diagonals
    exx_int = RegularGridInterpolator((np.arange(exx.shape[0]),
                                       np.arange(exx.shape[1])), exx)
    eyy_int = RegularGridInterpolator((np.arange(eyy.shape[0]),
                                       np.arange(eyy.shape[1])), eyy)
    exy_int = RegularGridInterpolator((np.arange(exy.shape[0]),
                                       np.arange(exy.shape[1])), exy)
    exx_diags = exx_int(interp_pts)
    eyy_diags = eyy_int(interp_pts)
    exy_diags = exy_int(interp_pts)

    # Build interpolators for the angle fields and compute the angles on the
    # provided diagonals
    theta_1_int = RegularGridInterpolator((np.arange(theta_1.shape[0]),
                                           np.arange(theta_1.shape[1])),
                                          theta_1)
    theta_2_int = RegularGridInterpolator((np.arange(theta_2.shape[0]),
                                           np.arange(theta_2.shape[1])),
                                          theta_2)
    theta_3_int = RegularGridInterpolator((np.arange(theta_3.shape[0]),
                                           np.arange(theta_3.shape[1])),
                                          theta_3)
    theta_1_diags = theta_1_int(interp_pts)
    theta_2_diags = theta_2_int(interp_pts)
    theta_3_diags = theta_3_int(interp_pts)

    # Build interpolators for the standard deviation fields and compute the
    # standard deviations on the provided diagonals
    sigma_1_int = RegularGridInterpolator((np.arange(sigma_1.shape[0]),
                                           np.arange(sigma_1.shape[1])),
                                          sigma_1)
    sigma_2_int = RegularGridInterpolator((np.arange(sigma_2.shape[0]),
                                           np.arange(sigma_2.shape[1])),
                                          sigma_2)
    sigma_3_int = RegularGridInterpolator((np.arange(sigma_3.shape[0]),
                                           np.arange(sigma_3.shape[1])),
                                          sigma_3)
    sigma_1_diags = sigma_1_int(interp_pts)
    sigma_2_diags = sigma_2_int(interp_pts)
    sigma_3_diags = sigma_3_int(interp_pts)

    # Build interpolators for the density field and compute the density on the
    # provided diagonals
    density_int = RegularGridInterpolator((np.arange(density.shape[0]),
                                           np.arange(density.shape[1])),
                                          density)
    density_diags = density_int(interp_pts)

    return (exx_diags, eyy_diags, exy_diags, theta_1_diags, theta_2_diags,
            theta_3_diags, sigma_1_diags, sigma_2_diags, sigma_3_diags,
            density_diags)


def calc_density(density_base: np.ndarray,
                 dens_min: float) -> np.ndarray:
    """Computes the density from the base image and the minimum density
    value.

    Args:
        density_base: The base image from which to calculate the density map.
        dens_min: The minimum density that the less exposed pixel in the image
            should have, between 0 and 1.

    Returns:
        The density map associated with the image.
    """

    d_max, d_min = density_base.max(), density_base.min()
    return 1 - (1 - dens_min) * (density_base - d_min) / (d_max - d_min)


def stress_diag_to_force(sxx: np.ndarray,
                         syy: np.ndarray,
                         sxy: np.ndarray,
                         img_h: int,
                         diag_len: int,
                         normals: np.ndarray,
                         cosines: np.ndarray,
                         scale: float,
                         thickness: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Computes the effort in the x and y directions from the calculated stress
    maps and the various other parameters of the test and the images.

    Args:
        sxx: The computed xx stress field, as an array.
        syy: The computed yy stress field, as an array.
        sxy: The computed xy stress field, as an array.
        img_h: The height of the original image in pixels, for normalization.
        diag_len: The number of points in a single diagonal.
        normals: A numpy array containing for each interpolation point the
            normalized coordinates of its normal along the interpolation line.
        cosines: A numpy array containing for each interpolation point the
            scaling factor to use for correcting the inclination of the
            interpolation line.
        scale: The mm/pixel ratio of the image, as a float.
        thickness: The thickness of the sample in mm, as a float.

    Returns:
        Two arrays containing for each diagonal the final computed effort value
        in the x and y directions, respectively.
    """

    # Normalize by the number of points in a digonal otherwise with fewer
    # points one cannot reach the target force with the same properties
    downscale_factor_h = diag_len / img_h

    # Integrate the stress over the diagonals to get the x and y efforts
    stress = np.stack((np.stack((sxx, sxy), axis=2),
                       np.stack((sxy, syy), axis=2)), axis=3)
    proj = (stress @ normals).squeeze() * scale * thickness / cosines
    sum_diags = np.sum(proj, axis=1)
    sum_diags /= downscale_factor_h

    return sum_diags[:, 0], sum_diags[:, 1]


def save_results(fit_vals: np.ndarray,
                 dest_file: Path,
                 order_coeffs: np.ndarray,
                 to_fit: np.ndarray,
                 extra_vals: np.ndarray,
                 index: int) -> None:
    """Writes the optimized parameter values to the indicated destination file.

    Args:
        fit_vals: Array containing the optimized parameter values from the
            least-squares fit.
        dest_file: Path to a .csv file where to store the output of the
            optimization.
        order_coeffs:
        to_fit: Array of flags indicating for each parameter of the model
            whether it was considered during optimization or if its value was
            fixed.
        extra_vals: Array containing the values of the parameters that were
            fixed during optimization.
        index: Index of the processed image in case only one image is being
            processed, otherwise leave to default.
    """

    # The labels of all the value to save for making the model
    labels = ("idx", "val1", "val2", "val3", "val4", "val5", "density_min",
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
    new_vals[labels[0]] = index

    for label, val in zip(labels[1:6], order_coeffs.tolist(), strict=True):
        new_vals[label] = val

    # Store all the other values for the material coefficients
    extra = iter(extra_vals.tolist())
    fitted = iter(fit_vals.tolist())
    for label, flag in zip(labels[6:], to_fit.tolist(), strict=True):
        if flag:
            new_vals[label] = next(fitted)
        else:
            new_vals[label] = next(extra)

    # Fuse the new results with the existing ones and save to a csv file
    results = pd.concat((results, new_vals.to_frame().transpose()),
                        ignore_index=True)
    results.to_csv(dest_file, columns=labels, index=False)
