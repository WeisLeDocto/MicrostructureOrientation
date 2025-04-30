# coding: utf-8

from collections.abc import Sequence
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, Bounds
from pathlib import Path
from tqdm.auto import tqdm
import sys

from compute_stress import compute_stress
from image_correlation import image_correlation


def error_diagonal(lib_path: Path,
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
    """"""

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

    sxx_int = RegularGridInterpolator((np.arange(sxx.shape[0]),
                                       np.arange(sxx.shape[1])), sxx)
    syy_int = RegularGridInterpolator((np.arange(syy.shape[0]),
                                       np.arange(syy.shape[1])), syy)
    sxy_int = RegularGridInterpolator((np.arange(sxy.shape[0]),
                                       np.arange(sxy.shape[1])), sxy)

    sxx_diags = sxx_int(interp_pts)
    syy_diags = syy_int(interp_pts)
    sxy_diags = sxy_int(interp_pts)
    stress = np.stack((np.stack((sxx_diags, sxy_diags), axis=2),
                       np.stack((sxy_diags, syy_diags), axis=2)), axis=3)
    proj = (stress @ normals).squeeze() * scale * thickness / cosines
    sum_diags = np.sum(proj, axis=1)

    return np.sum(np.sqrt(np.power(sum_diags[:, 0] - effort_x, 2) +
                          np.power(sum_diags[:, 1] - effort_y, 2)),
                  axis=None) / interp_pts.shape[0]


def error_all_diagonals(lib_path: Path,
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
                        effort_y: float) -> None:
    """"""

    exx_int = RegularGridInterpolator((np.arange(exx.shape[0]),
                                       np.arange(exx.shape[1])), exx)
    eyy_int = RegularGridInterpolator((np.arange(eyy.shape[0]),
                                       np.arange(eyy.shape[1])), eyy)
    exy_int = RegularGridInterpolator((np.arange(exy.shape[0]),
                                       np.arange(exy.shape[1])), exy)
    exx_diags = exx_int(interp_pts)
    eyy_diags = eyy_int(interp_pts)
    exy_diags = exy_int(interp_pts)

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

    density_int = RegularGridInterpolator((np.arange(density.shape[0]),
                                           np.arange(density.shape[1])),
                                          density)
    density_diags = density_int(interp_pts)

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

    stress = np.stack((np.stack((sxx, sxy), axis=2),
                       np.stack((sxy, syy), axis=2)), axis=3)
    proj = (stress @ normals).squeeze() * scale * thickness / cosines
    sum_diags = np.sum(proj, axis=1)

    return np.sum(np.sqrt(np.power(sum_diags[:, 0] - effort_x, 2) +
                          np.power(sum_diags[:, 1] - effort_y, 2)),
                  axis=None) / interp_pts.shape[0]


def error_diagonals(lib_path: Path,
                    interp_strain: bool,
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
    """"""

    error = 0.0
    err_func = error_all_diagonals if interp_strain else error_diagonal
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
                                     position=0,
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

    print("\n")
    print(error)
    return error


def wrapper(x: np.ndarray,
            to_fit: np.ndarray,
            extra_vals: np.ndarray,
            interp_strain,
            val1,
            val2,
            val3,
            val4,
            val5,
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
    """"""

    lambdas = np.empty(16, dtype=np.float64)
    x = iter(x)
    extra_vals = iter(extra_vals)

    dens_min = next(x)

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

    print(dens_min, lambda_h,
          lambda_11, lambda_21, lambda_51,
          lambda_12, lambda_22, lambda_52,
          lambda_13, lambda_23, lambda_53,
          lambda_14, lambda_24, lambda_54,
          lambda_15, lambda_25, lambda_55)

    d_max, d_min = density_base.max(), density_base.min()
    density = 1 - (1 - dens_min) * (density_base - d_min) / (d_max - d_min)

    return error_diagonals(lib_path,
                           interp_strain,
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
                       diagonal_downscaling: int) -> None:
    """"""

    exxs, eyys, exys = zip(*(image_correlation(ref_img, def_img)
                             for def_img in def_images))

    sigma_1 = gauss_fit[:, :, 0]
    sigma_2 = np.nan_to_num(gauss_fit[:, :, 2])
    sigma_3 = np.nan_to_num(gauss_fit[:, :, 4])

    theta_1 = peaks[:, :, 0]
    theta_2 = np.nan_to_num(peaks[:, :, 1])
    theta_3 = np.nan_to_num(peaks[:, :, 2])

    interp_pts = np.empty((ref_img.shape[1], nb_interp_diag, 2),
                          dtype=np.float64)
    for j in range(interp_pts.shape[0]):
        interp_pts[j, :, 1] = np.linspace(j, ref_img.shape[1] - 1 - j,
                                          nb_interp_diag)
        interp_pts[j, :, 0] = np.linspace(0, ref_img.shape[0] - 1,
                                          nb_interp_diag)

    interp_pts = interp_pts[::diagonal_downscaling]

    normals = np.zeros((ref_img.shape[1], nb_interp_diag, 2), dtype=np.float64)
    for i in range(normals.shape[0]):
        normals[i] = np.array((1.0, (ref_img.shape[1] - 2 * i - 1) /
                               ref_img.shape[0]),
                              dtype=np.float64)
        normals[i] /= np.linalg.norm(normals[i], axis=1)[:, np.newaxis]

    normals = normals[::diagonal_downscaling]

    cosines = normals @ np.array((1.0, 0.0), dtype=np.float64)
    normals = normals[..., np.newaxis]
    cosines = cosines[..., np.newaxis]

    nb_max = 17
    low_bounds = np.array((1.0e-5,) * nb_max)
    low_bounds[0] = 0.0
    low_bounds[1] = 1.0
    high_bounds = np.array((np.inf,) * nb_max)
    high_bounds[0] = 0.99

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

    fit = least_squares(wrapper, x0, bounds=bounds, x_scale=x_scale,
                        kwargs={'to_fit': to_fit,
                                'extra_vals': extra_vals,
                                'interp_strain': interp_strain,
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
    print(fit.x)


if __name__ == "__main__":

    ref_img_pth = Path("/home/weis/Desktop/HDR/7LX1_2/hdr/4_2414.npy")
    ref_img = np.load(ref_img_pth)

    lib_path = Path("/home/weis/Codes/tfel/build/bidouillage/Kelvin/"
                    "kelvin_lib.so")

    dens_base_pth = Path("/home/weis/Desktop/HDR/7LX1_2/images/"
                         "2424_300_35.npy")
    density_base = np.load(dens_base_pth)
    roi_y = slice(1480, 2897, 1)
    roi_x = slice(698, 1898, 1)
    density_base = density_base[roi_x, roi_y]

    gauss_fit_path = Path("/home/weis/Desktop/HDR/7LX1_2/fit/4_2414.npy")
    gauss_fit = np.load(gauss_fit_path)

    peaks_path = Path("/home/weis/Desktop/HDR/7LX1_2/peaks/4_2414.npz")
    peaks = np.radians(np.load(peaks_path)['arr_0'])

    # 7LX1_2 14
    x0 = np.concatenate((np.array((0.4821446796378303,
                                   26.99900699137642),
                                  dtype=np.float64),
                         np.array((0.6514774582289321,
                                   1.841715651404761,
                                   0.8314353094200984),
                                  dtype=np.float64),
                         np.tile(np.array((0.11400011168738121,
                                           0.00021413006227584167,
                                           0.2536989285296272),
                                          dtype=np.float64),
                                 4)), axis=0)

    def_images_paths = (Path("/home/weis/Desktop/HDR/7LX1_2/hdr/14_4196.npy"),)
    def_images = tuple(np.load(img) for img in def_images_paths)

    efforts_x = (0.868,)
    efforts_y = (0.0,)

    order_coeffs = np.array((1.0, 0.0, 1.0, 0.0, 0.0))

    scale = 0.01
    thickness = 0.54

    nb_interp_diag = ref_img.shape[0]
    interp_strain = True
    diagonal_downscaling = 20

    optimize_diagonals(lib_path,
                       ref_img,
                       density_base,
                       gauss_fit,
                       peaks,
                       x0,
                       def_images,
                       efforts_x,
                       efforts_y,
                       order_coeffs,
                       scale,
                       thickness,
                       nb_interp_diag,
                       interp_strain,
                       diagonal_downscaling)
