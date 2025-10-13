# coding: utf-8

import numpy as np
from math import log, exp, pow, sqrt, atan2, cos
from typing import Sequence
from scipy.linalg import expm, fractional_matrix_power
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import sys
from itertools import batched


class EzzBuf:
    ezz = 0.0


def kelvin_integrated_tensor(lambda_h: float, 
                             lambda_1: float, 
                             lambda_2: float, 
                             lambda_3: float, 
                             lambda_4: float, 
                             lambda_5: float, 
                             sigma: float,
                             m: float) -> np.ndarray:
    """"""
  
    integrated = np.zeros((6, 6), dtype=np.float64)

    # Pre-compute some terms to optimize speed and avoid redundancy
    lamh_m = pow(lambda_h, m)
    lam1_m = pow(lambda_1, m)
    lam2_m = pow(lambda_2, m)
    lam3_m = pow(lambda_3, m)
    lam4_m = pow(lambda_4, m)
    lam5_m = pow(lambda_5, m)
  
    sigma_sq = sigma * sigma
  
    exp_2s = exp(2.0 * sigma_sq)
    exp_6s = exp_2s * exp_2s * exp_2s
    exp_8s = exp_2s * exp_6s
    exp_neg2s = 1.0 / exp_2s
    exp_neg8s = 1.0 / exp_8s
  
    term_a = 11.0 * lam1_m + 9.0 * lam2_m + 12.0 * lam5_m + 16.0 * lamh_m
    term_b = 12.0 * (lam1_m - lam2_m)
    term_c = 9.0 * lam1_m + 3.0 * lam2_m - 12.0 * lam5_m
    term_d = 7.0 * lam1_m - 3.0 * lam2_m + 12.0 * lam5_m - 16.0 * lamh_m
    term_e = lam1_m + 3.0 * lam2_m - 4.0 * lamh_m
    term_f = 3.0 * lam1_m + lam2_m + 4.0 * lam5_m
    term_g = 3.0 * lam1_m - 3.0 * lam2_m
    term_h = lam3_m + lam4_m
    term_j = lam3_m - lam4_m

    # Compute the terms on the upper half of the tensor
    integrated[0, 0] = 1.0 / 48.0 * (term_a + term_b * exp_neg2s + term_c
                                     * exp_neg8s)
    integrated[0, 1] = -1.0 / 48.0 * (term_d + term_c * exp_neg8s)
    integrated[0, 2] = -1.0 / 12.0 * (term_e + term_g * exp_neg2s)
    integrated[1, 1] = 1.0 / 48.0 * (term_a - term_b * exp_neg2s + term_c
                                     * exp_neg8s)
    integrated[1, 2] = -1.0 / 12.0 * (term_e - term_g * exp_neg2s)
    integrated[2, 2] = (1.0 / 6.0 * lam1_m + 1.0 / 2.0 * lam2_m + 1.0 / 3.0
                        * lamh_m)
    integrated[3, 3] = 1.0 / 8.0 * (term_f + (-3.0 * lam1_m - lam2_m + 4.0
                                              * lam5_m) * exp_neg8s)
    integrated[4, 4] = 0.5 * (term_h - term_j * exp_neg2s)
    integrated[5, 5] = 0.5 * (term_h + term_j * exp_neg2s)

    # The tensor is symmetrical, so the lower half terms can just be copied
    integrated[1, 0] = integrated[0, 1]
    integrated[2, 0] = integrated[0, 2]
    integrated[2, 1] = integrated[1, 2]
  
    return integrated


def zero_m_approximation(lambda_h: float, 
                         lambda_1: float, 
                         lambda_2: float, 
                         lambda_3: float, 
                         lambda_4: float, 
                         lambda_5: float, 
                         sigma: float,
                         m: float) -> np.ndarray:
    """"""

    integrated = np.zeros((6, 6), dtype=np.float64)

    # Pre-compute some terms to optimize speed and avoid redundancy
    lamh_l = log(lambda_h)
    lam1_l = log(lambda_1)
    lam2_l = log(lambda_2)
    lam3_l = log(lambda_3)
    lam4_l = log(lambda_4)
    lam5_l = log(lambda_5)
  
    sigma_sq = sigma * sigma
  
    exp_2s = exp(2.0 * sigma_sq)
    exp_4s = exp_2s * exp_2s
    exp_6s = exp_2s * exp_4s
    exp_8s = exp_4s * exp_4s
    exp_10s = exp_8s * exp_2s
    exp_12s = exp_8s * exp_4s
    exp_14s = exp_12s * exp_2s
    exp_16s = exp_8s * exp_8s
  
    exp_neg2s = 1.0 / exp_2s
    exp_neg4s = 1.0 / exp_4s
    exp_neg8s = 1.0 / exp_8s
    exp_neg10s = 1.0 / exp_10s
    exp_neg16s = 1.0 / exp_16s
  
    term_a = 11.0 * lam1_l + 9.0 * lam2_l + 12.0 * lam5_l + 16.0 * lamh_l
    term_b = 12.0 * (lam1_l - lam2_l)
    term_c = 9.0 * lam1_l + 3.0 * lam2_l - 12.0 * lam5_l
    term_d = 7.0 * lam1_l - 3.0 * lam2_l + 12.0 * lam5_l - 16.0 * lamh_l
    term_e = lam1_l + 3.0 * lam2_l - 4.0 * lamh_l
    term_f = 3.0 * lam1_l + lam2_l + 4.0 * lam5_l
    term_g = 3.0 * lam1_l - 3.0 * lam2_l
    term_h = lam3_l + lam4_l
    term_j = lam3_l - lam4_l
  
    lam1_sq = lam1_l * lam1_l
    lam2_sq = lam2_l * lam2_l
    lam5_sq = lam5_l * lam5_l
    lam1_lam2 = lam1_l * lam2_l
    lam1_lam5 = lam1_l * lam5_l
    lam2_lam5 = lam2_l * lam5_l
    cross_term = lam1_sq - 2.0 * lam1_lam2 + lam2_sq
    z_cross = lam3_l * lam3_l - 2.0 * lam3_l * lam4_l + lam4_l * lam4_l
    z_term = z_cross * m * (exp_4s - 1.0) * exp_neg4s
  
    cross_term_m = cross_term * m
    quad_term1 = (19.0 * lam1_sq - 14.0 * lam1_lam2 + 11.0 * lam2_sq - 24.0
                  * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq)
    quad_term2 = (3.0 * lam1_sq - 2.0 * lam1_lam2 - lam2_sq - 4.0 * lam1_lam5
                  + 4.0 * lam2_lam5)
    quad_term3 = (9.0 * lam1_sq + 6.0 * lam1_lam2 + lam2_sq - 24.0 * lam1_lam5
                  - 8.0 * lam2_lam5 + 16.0 * lam5_sq)
    quad_term4 = (15.0 * lam1_sq - 6.0 * lam1_lam2 + 7.0 * lam2_sq - 24.0
                  * lam1_lam5 - 8.0 * lam2_lam5 + 16.0 * lam5_sq)

    # Compute the terms on the upper half of the tensor
    integrated[0, 0] = (1.0 / 48.0 * (term_a * exp_8s + term_b * exp_6s
                                      + term_c) * exp_neg8s)
    integrated[0, 1] = -1.0 / 48.0 * (term_d * exp_8s + term_c) * exp_neg8s
    integrated[0, 2] = -1.0 / 12.0 * (term_e * exp_2s + term_g) * exp_neg2s
    integrated[1, 1] = (1.0 / 48.0 * (term_a * exp_8s - term_b * exp_6s
                                      + term_c) * exp_neg8s)
    integrated[1, 2] = -1.0 / 12.0 * (term_e * exp_2s - term_g) * exp_neg2s
    integrated[2, 2] = (1.0 / 6.0 * lam1_l + 1.0 / 2.0 * lam2_l + 1.0 / 3.0
                        * lamh_l)
    integrated[3, 3] = (1.0 / 8.0 * (term_f * exp_8s - 3.0 * lam1_l - lam2_l
                                     + 4.0 * lam5_l) * exp_neg8s)
    integrated[4, 4] = 0.5 * (term_h * exp_2s - term_j) * exp_neg2s
    integrated[5, 5] = 0.5 * (term_h * exp_2s + term_j) * exp_neg2s

    # The result is a sum of two terms, computing the second one here
    integrated[0, 0] += (1.0 / 256.0 * m * (quad_term1 * exp_16s + 4.0
                                            * quad_term2 * exp_14s - 16.0
                                            * cross_term * exp_12s + 6.0
                                            * cross_term * exp_8s - 4.0
                                            * quad_term2 * exp_6s - quad_term3)
                         * exp_neg16s)
    integrated[0, 1] += (-1.0 / 256.0 * m * ((11.0 * lam1_sq + 2.0 * lam1_lam2
                                              + 3.0 * lam2_sq - 24.0
                                              * lam1_lam5 - 8.0 * lam2_lam5
                                              + 16.0 * lam5_sq) * exp_16s
                                             - 8.0 * cross_term * exp_12s + 6.0
                                             * cross_term * exp_8s
                                             - quad_term3) * exp_neg16s)
    integrated[0, 2] += (-1.0 / 64.0 * m * (2.0 * cross_term * exp_10s +
                                            (3.0 * lam1_sq - 2.0 * lam1_lam2
                                             - lam2_sq - 4.0 * lam1_lam5 + 4.0
                                             * lam2_lam5) * exp_8s - 2.0
                                            * cross_term * exp_6s -
                                            (3.0 * lam1_sq - 2.0 * lam1_lam2
                                             - lam2_sq - 4.0 * lam1_lam5 + 4.0
                                             * lam2_lam5)) * exp_neg10s)
    integrated[1, 1] += (1.0 / 256.0 * m * (quad_term1 * exp_16s - 4.0
                                            * quad_term2 * exp_14s - 16.0
                                            * cross_term * exp_12s + 6.0
                                            * cross_term * exp_8s + 4.0
                                            * quad_term2 * exp_6s - quad_term3)
                         * exp_neg16s)
    integrated[1, 2] += (-1.0 / 64.0 * m * (2.0 * cross_term * exp_10s -
                                            (3.0 * lam1_sq - 2.0 * lam1_lam2
                                             - lam2_sq - 4.0 * lam1_lam5 + 4.0
                                             * lam2_lam5) * exp_8s - 2.0
                                            * cross_term * exp_6s +
                                            (3.0 * lam1_sq - 2.0 * lam1_lam2
                                             - lam2_sq - 4.0 * lam1_lam5 + 4.0
                                             * lam2_lam5)) * exp_neg10s)
    integrated[2, 2] += 1.0 / 16.0 * cross_term_m * (exp_4s - 1.0) * exp_neg4s
    integrated[3, 3] += (1.0 / 128.0 * m * (quad_term4 * exp_16s - 6.0
                                            * cross_term * exp_8s - quad_term3)
                         * exp_neg16s)
    integrated[4, 4] += 0.125 * z_term
    integrated[5, 5] += 0.125 * z_term

    # The tensor is symmetrical, so the lower half terms can just be copied
    integrated[1, 0] = integrated[0, 1]
    integrated[2, 0] = integrated[0, 2]
    integrated[2, 1] = integrated[1, 2]
  
    return integrated


def rotate(tensor: np.ndarray, angle: float) -> np.ndarray:
    """"""

    # Build the rotation matrix in R3
    rot_mat = np.array(((np.cos(angle), -np.sin(angle), 0), 
                        (np.sin(angle), np.cos(angle), 0), 
                        (0, 0, 1)), dtype=np.float64)

    # Build the rotation tensor from the rotation matrix
    rot_ten = np.zeros((6, 6), dtype=np.float64)
  
    rot_ten[0, 0] = rot_mat[0, 0] * rot_mat[0, 0]
    rot_ten[0, 1] = rot_mat[1, 0] * rot_mat[1, 0]
    rot_ten[0, 2] = rot_mat[2, 0] * rot_mat[2, 0]
    rot_ten[0, 3] = sqrt(2) * rot_mat[0, 0] * rot_mat[1, 0]
    rot_ten[0, 4] = sqrt(2) * rot_mat[0, 0] * rot_mat[2, 0]
    rot_ten[0, 5] = sqrt(2) * rot_mat[1, 0] * rot_mat[2, 0]

    rot_ten[1, 0] = rot_mat[0, 1] * rot_mat[0, 1]
    rot_ten[1, 1] = rot_mat[1, 1] * rot_mat[1, 1]
    rot_ten[1, 2] = rot_mat[2, 1] * rot_mat[2, 1]
    rot_ten[1, 3] = sqrt(2) * rot_mat[0, 1] * rot_mat[1, 1]
    rot_ten[1, 4] = sqrt(2) * rot_mat[0, 1] * rot_mat[2, 1]
    rot_ten[1, 5] = sqrt(2) * rot_mat[1, 1] * rot_mat[2, 1]

    rot_ten[2, 0] = rot_mat[0, 2] * rot_mat[0, 2]
    rot_ten[2, 1] = rot_mat[1, 2] * rot_mat[1, 2]
    rot_ten[2, 2] = rot_mat[2, 2] * rot_mat[2, 2]
    rot_ten[2, 3] = sqrt(2) * rot_mat[0, 2] * rot_mat[1, 2]
    rot_ten[2, 4] = sqrt(2) * rot_mat[0, 2] * rot_mat[2, 2]
    rot_ten[2, 5] = sqrt(2) * rot_mat[1, 2] * rot_mat[2, 2]

    rot_ten[3, 0] = sqrt(2) * rot_mat[0, 0] * rot_mat[0, 1]
    rot_ten[3, 1] = sqrt(2) * rot_mat[1, 0] * rot_mat[1, 1]
    rot_ten[3, 2] = sqrt(2) * rot_mat[2, 0] * rot_mat[2, 1]
    rot_ten[3, 3] = (rot_mat[0, 0] * rot_mat[1, 1] + rot_mat[0, 1] 
                     * rot_mat[1, 0])
    rot_ten[3, 4] = (rot_mat[0, 0] * rot_mat[2, 1] + rot_mat[0, 1] 
                     * rot_mat[2, 0])
    rot_ten[3, 5] = (rot_mat[1, 0] * rot_mat[2, 1] + rot_mat[1, 1] 
                     * rot_mat[2, 0])

    rot_ten[4, 0] = sqrt(2) * rot_mat[0, 0] * rot_mat[0, 2]
    rot_ten[4, 1] = sqrt(2) * rot_mat[1, 0] * rot_mat[1, 2]
    rot_ten[4, 2] = sqrt(2) * rot_mat[2, 0] * rot_mat[2, 2]
    rot_ten[4, 3] = (rot_mat[0, 0] * rot_mat[1, 2] + rot_mat[0, 2] 
                     * rot_mat[1, 0])
    rot_ten[4, 4] = (rot_mat[0, 0] * rot_mat[2, 2] + rot_mat[0, 2] 
                     * rot_mat[2, 0])
    rot_ten[4, 5] = (rot_mat[1, 0] * rot_mat[2, 2] + rot_mat[1, 2] 
                     * rot_mat[2, 0])

    rot_ten[5, 0] = sqrt(2) * rot_mat[0, 1] * rot_mat[0, 2]
    rot_ten[5, 1] = sqrt(2) * rot_mat[1, 1] * rot_mat[1, 2]
    rot_ten[5, 2] = sqrt(2) * rot_mat[2, 1] * rot_mat[2, 2]
    rot_ten[5, 3] = (rot_mat[0, 1] * rot_mat[1, 2] + rot_mat[0, 2] 
                     * rot_mat[1, 1])
    rot_ten[5, 4] = (rot_mat[0, 1] * rot_mat[2, 2] + rot_mat[0, 2] 
                     * rot_mat[2, 1])
    rot_ten[5, 5] = (rot_mat[1, 1] * rot_mat[2, 2] + rot_mat[1, 2] 
                     * rot_mat[2, 1])

    # Finally, apply the rotation tensor
    return rot_ten.transpose() @ tensor @ rot_ten


def final_stiffness(stiffness: np.ndarray,
                    strain: np.ndarray,
                    vals: Sequence[float],
                    valid_layers: Sequence[int],
                    valid_orders: Sequence[int]) -> np.ndarray:
    """"""

    # Declare variables
    stiff_tot = np.zeros((6, 6), dtype=np.float64)

    # Iterate over all the valid layers
    for i in valid_layers:

        # Iterate over all the valid orders
        for j in valid_orders:

            # For each order greater than 1, there is a multiplicative factor
            # dependent on the strain and stiffness
            factor = strain.transpose() @ stiffness[i][j] @ strain
            factor = np.squeeze(factor)

            # The total equivalent stiffness is the sum of the ones for 
            # each order
            stiff_tot += vals[j] * pow(factor, j) * stiffness[i][j]
  
    return stiff_tot / len(valid_layers)


def calc_ezz_plane_stress(stiffness: np.ndarray,
                          strain: np.ndarray,
                          vals: Sequence[float],
                          valid_layers: Sequence[int],
                          valid_orders: Sequence[int],
                          stop_crit: float,
                          max_iter: int) -> float:
    """"""

    # Iterate until reaching a solution or until reaching max iterations
    n = 0
    while n < max_iter:
      n += 1

      # Calculate stress
      stiff_tot = final_stiffness(stiffness,
                                  strain,
                                  vals,
                                  valid_layers,
                                  valid_orders)
      stress = stiff_tot @ strain

      # Stop here if the stress is already close enough to 0
      if (abs(stress[2, 0]) < stop_crit *
          max(abs(stress[0, 0]), abs(stress[1, 0]), abs(stress[3, 0]))):
        return strain[2, 0]

      # Use a fixed-point method to compute the optimal strain value
      if (abs(stiff_tot[2, 2]) > 0.001 *
          min(abs(stiff_tot[2, 0]),
              abs(stiff_tot[2, 1]),
              abs(stiff_tot[2, 3]))):
          strain[2, 0] = -((stiff_tot[2, 0] * strain[0, 0] +
                            stiff_tot[2, 1] * strain[1, 0] +
                            sqrt(2) * stiff_tot[2, 3] * strain[3, 0])
                           / stiff_tot[2, 2])
      # In case the stiffness associated with zz is zero, nothing can be done
      else:
        return strain[2, 0]
  
    return strain[2, 0]


def calc_stress(exx: float,
                eyy: float,
                exy: float,
                m_vals: Sequence[float],
                lamh: float,
                lam1: Sequence[float],
                lam2: Sequence[float],
                lam3: Sequence[float],
                lam4: Sequence[float],
                lam5: Sequence[float],
                vals: Sequence[float],
                theta: Sequence[float],
                sigstd: Sequence[float],
                density: float) -> tuple[tuple[float, float, float], float]:
    """"""

    stiffness = np.zeros((3, 5, 6, 6), dtype=np.float64)

    valid_orders = tuple(i for i in range(5) if vals[i] > 1.0e-12)
    valid_layers = tuple(i for i in range(3) if (sigstd[i] > 0.01) or (i < 1))

    for i in valid_layers:
        for j in valid_orders:
            if abs(m_vals[i]) < 0.01:
                homogenized = expm(zero_m_approximation(
                    lamh, lam1[j], lam2[j], lam3[j], lam4[j], lam5[j],
                    sigstd[i], m_vals[i]))
            else:
                homogenized = fractional_matrix_power(kelvin_integrated_tensor(
                    lamh, lam1[j], lam2[j], lam3[j], lam4[j], lam5[j],
                    sigstd[i], m_vals[i]), 1.0 / m_vals[i])

            stiffness[i][j] = rotate(homogenized, theta[i])

    strain_3d = np.array(((exx,), (eyy,), (0.0,), (exy,), (0.0,), (0.0,)), 
                         dtype=np.float64)
    strain_3d[2, 0] = calc_ezz_plane_stress(stiffness, strain_3d, vals, 
                                            valid_layers, valid_orders,
                                            0.001, 100)
    sig_3d = density * final_stiffness(stiffness, strain_3d, vals,
                                       valid_layers, valid_orders) @ strain_3d

    return (sig_3d[0, 0], sig_3d[1, 0], sig_3d[3, 0]), strain_3d[2, 0]


def least_square_wrapper(x: np.ndarray,
                         target_stress: np.ndarray,
                         m_vals: Sequence[float],
                         lamh: float,
                         lam1: Sequence[float],
                         lam2: Sequence[float],
                         lam3: Sequence[float],
                         lam4: Sequence[float],
                         lam5: Sequence[float],
                         vals: Sequence[float],
                         theta: Sequence[float],
                         sigstd: Sequence[float],
                         density: float
                         ) -> float:
    """"""

    exx, eyy, exy = x
    txx, tyy, txy = target_stress
    (sxx, syy, sxy), ezz = calc_stress(exx,
                                       eyy,
                                       exy,
                                       m_vals,
                                       lamh,
                                       lam1,
                                       lam2,
                                       lam3,
                                       lam4,
                                       lam5,
                                       vals,
                                       theta,
                                       sigstd,
                                       density)
    EzzBuf.ezz = ezz
    return (txx - sxx) ** 2 + (tyy - syy) ** 2 + (txy - sxy) ** 2


if __name__ == '__main__':

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    diff = (expm(zero_m_approximation(100.0,
                                      5.0,
                                      3.0,
                                      3.0,
                                      2.0,
                                      2.0,
                                      1.0,
                                      -0.01)) -
            fractional_matrix_power(kelvin_integrated_tensor(100.0,
                                                             5.0,
                                                             3.0,
                                                             3.0,
                                                             2.0,
                                                             2.0,
                                                             1.0,
                                                             -0.01),
                                    1 / -0.01))

    baseline = fractional_matrix_power(kelvin_integrated_tensor(100.0,
                                                                5.0,
                                                                3.0,
                                                                3.0,
                                                                2.0,
                                                                2.0,
                                                                1.0,
                                                                -0.01),
                                       1 / -0.01)
    nb_no_z = np.count_nonzero(baseline)
    error = np.sqrt(np.sum(np.power(diff, 2)) / nb_no_z)
    avg = np.sum(np.abs(baseline)) / nb_no_z
    print(error)
    print(avg)

    diff = (expm(zero_m_approximation(100.0,
                                      5.0,
                                      3.0,
                                      3.0,
                                      2.0,
                                      2.0,
                                      1.0,
                                      0.01)) -
            fractional_matrix_power(kelvin_integrated_tensor(100.0,
                                                             5.0,
                                                             3.0,
                                                             3.0,
                                                             2.0,
                                                             2.0,
                                                             1.0,
                                                             0.01),
                                    1 / 0.01))

    baseline = fractional_matrix_power(kelvin_integrated_tensor(100.0,
                                                                5.0,
                                                                3.0,
                                                                3.0,
                                                                2.0,
                                                                2.0,
                                                                1.0,
                                                                0.01),
                                       1 / 0.01)
    nb_no_z = np.count_nonzero(baseline)
    error = np.sqrt(np.sum(np.power(diff, 2)) / nb_no_z)
    avg = np.sum(np.abs(baseline)) / nb_no_z
    print(error)
    print(avg)

    angles = (0.0, 0.0, 0.0)
    sigma = (0.5, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (1.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress = (0.0, 0.0, 1.0)

    if (abs(stress[2]) < 0.001
            and abs(stress[0] - stress[1]) < 0.001):
        m = (0.0, 0.0, 0.0)
    else:
        m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                               stress[0] -
                                               stress[1])))
                  for theta in angles)

    strain = least_squares(least_square_wrapper,
                           x0=(0.0, 0.0, 0.0),
                           kwargs={'target_stress': stress,
                                   'm_vals': m,
                                   'lamh': lh,
                                   'lam1': l1,
                                   'lam2': l2,
                                   'lam3': l3,
                                   'lam4': l4,
                                   'lam5': l5,
                                   'vals': coeffs,
                                   'theta': angles,
                                   'sigstd': sigma,
                                   'density': dens}).x

    (_, _, sxy_0), _ = calc_stress(strain[0],
                                   strain[1],
                                   strain[2],
                                   m,
                                   lh,
                                   l1,
                                   l2,
                                   l3,
                                   l3,
                                   l5,
                                   coeffs,
                                   angles,
                                   sigma,
                                   dens)

    plan = ((0.8, 1.0, 1.0),
            (1.2, 1.0, 1.0),
            (1.0, 0.8, 1.0),
            (1.0, 1.2, 1.0),
            (1.0, 1.0, 0.8),
            (1.0, 1.0, 1.2))

    res = list()

    for (f1, f2, f5) in plan:
        lam1 = tuple(e * f1 for e in l1)
        lam2 = tuple(e * f2 for e in l2)
        lam5 = tuple(e * f5 for e in l5)

        (_, _, sxy), _ = calc_stress(strain[0],
                                     strain[1],
                                     strain[2],
                                     m,
                                     lh,
                                     lam1,
                                     lam2,
                                     lam2,
                                     lam5,
                                     lam5,
                                     coeffs,
                                     angles,
                                     sigma,
                                     dens)
        res.append((sxy - sxy_0) / sxy_0)

    fig = plt.figure(figsize=(17, 3))
    ax = fig.add_subplot(1, 3, 1)

    ax.vlines(0, -0.225, 1.225, color='k')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.yaxis.set_tick_params(length=0, labelsize=12)
    ax.set_title(r'Relative variation in $\tau_{xy}$', fontsize=12)
    ax.set_xticks((-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15))

    ax.barh((0.5, 0.5, 0.0, 0.0, 1.0, 1.0),
            res, align='center', height=0.45,
            color=('coral', 'deepskyblue') * 3,
            tick_label=(r'$\lambda_1$', r'$\lambda_1$',
                        r'$\lambda_2$', r'$\lambda_2$',
                        r'$\lambda_5$', r'$\lambda_5$'))

    ax.set_xlim((-max(*map(abs, ax.get_xlim())),
                 max(*map(abs, ax.get_xlim()))))

    stress = (1.0, 0.0, 0.0)

    if (abs(stress[2]) < 0.001
            and abs(stress[0] - stress[1]) < 0.001):
        m = (0.0, 0.0, 0.0)
    else:
        m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                               stress[0] -
                                               stress[1])))
                  for theta in angles)

    strain = least_squares(least_square_wrapper,
                           x0=(0.0, 0.0, 0.0),
                           kwargs={'target_stress': stress,
                                   'm_vals': m,
                                   'lamh': lh,
                                   'lam1': l1,
                                   'lam2': l2,
                                   'lam3': l3,
                                   'lam4': l4,
                                   'lam5': l5,
                                   'vals': coeffs,
                                   'theta': angles,
                                   'sigstd': sigma,
                                   'density': dens}).x

    (sxx_0, _, _), _ = calc_stress(strain[0],
                                   strain[1],
                                   strain[2],
                                   m,
                                   lh,
                                   l1,
                                   l2,
                                   l3,
                                   l3,
                                   l5,
                                   coeffs,
                                   angles,
                                   sigma,
                                   dens)

    plan = ((0.8, 1.0, 1.0),
            (1.2, 1.0, 1.0),
            (1.0, 0.8, 1.0),
            (1.0, 1.2, 1.0),
            (1.0, 1.0, 0.8),
            (1.0, 1.0, 1.2))

    res = list()

    for (f1, f2, f5) in plan:
        lam1 = tuple(e * f1 for e in l1)
        lam2 = tuple(e * f2 for e in l2)
        lam5 = tuple(e * f5 for e in l5)

        (sxx, _, _), _ = calc_stress(strain[0],
                                     strain[1],
                                     strain[2],
                                     m,
                                     lh,
                                     lam1,
                                     lam2,
                                     lam2,
                                     lam5,
                                     lam5,
                                     coeffs,
                                     angles,
                                     sigma,
                                     dens)
        res.append((sxx - sxx_0) / sxx_0)

    ax = fig.add_subplot(1, 3, 2)

    ax.vlines(0, -0.225, 1.225, color='k')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.yaxis.set_tick_params(length=0, labelsize=12)
    ax.set_title(r'Relative variation in $\tau_{xx}$ ($0^\circ$)', fontsize=12)
    ax.set_xticks((-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15))
    ax.set_xlim((-0.11, 0.11))

    ax.barh((1.0, 1.0, 0.0, 0.0, 0.5, 0.5),
            res, align='center', height=0.45,
            color=('coral', 'deepskyblue') * 3,
            tick_label=(r'$\lambda_1$', r'$\lambda_1$',
                        r'$\lambda_2$', r'$\lambda_2$',
                        r'$\lambda_5$', r'$\lambda_5$'))

    angles = (np.pi / 2, 0.0, 0.0)

    if (abs(stress[2]) < 0.001
            and abs(stress[0] - stress[1]) < 0.001):
        m = (0.0, 0.0, 0.0)
    else:
        m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                               stress[0] -
                                               stress[1])))
                  for theta in angles)

    strain = least_squares(least_square_wrapper,
                           x0=(0.0, 0.0, 0.0),
                           kwargs={'target_stress': stress,
                                   'm_vals': m,
                                   'lamh': lh,
                                   'lam1': l1,
                                   'lam2': l2,
                                   'lam3': l3,
                                   'lam4': l4,
                                   'lam5': l5,
                                   'vals': coeffs,
                                   'theta': angles,
                                   'sigstd': sigma,
                                   'density': dens}).x

    (sxx_0, _, _), _ = calc_stress(strain[0],
                                   strain[1],
                                   strain[2],
                                   m,
                                   lh,
                                   l1,
                                   l2,
                                   l3,
                                   l3,
                                   l5,
                                   coeffs,
                                   angles,
                                   sigma,
                                   dens)

    plan = ((0.8, 1.0, 1.0),
            (1.2, 1.0, 1.0),
            (1.0, 0.8, 1.0),
            (1.0, 1.2, 1.0),
            (1.0, 1.0, 0.8),
            (1.0, 1.0, 1.2))

    res = list()

    for (f1, f2, f5) in plan:
        lam1 = tuple(e * f1 for e in l1)
        lam2 = tuple(e * f2 for e in l2)
        lam5 = tuple(e * f5 for e in l5)

        (sxx, _, _), _ = calc_stress(strain[0],
                                     strain[1],
                                     strain[2],
                                     m,
                                     lh,
                                     lam1,
                                     lam2,
                                     lam2,
                                     lam5,
                                     lam5,
                                     coeffs,
                                     angles,
                                     sigma,
                                     dens)
        res.append((sxx - sxx_0) / sxx_0)

    ax = fig.add_subplot(1, 3, 3)

    ax.vlines(0, -0.225, 1.225, color='k')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.yaxis.set_tick_params(length=0, labelsize=12)
    ax.set_title(r'Relative variation in $\tau_{xx}$ ($90^\circ$)',
                 fontsize=12)
    ax.set_xticks((-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15))

    ax.barh((0.0, 0.0, 1.0, 1.0, 0.5, 0.5),
            res, align='center', height=0.45,
            color=('coral', 'deepskyblue') * 3,
            tick_label=(r'$\lambda_1$', r'$\lambda_1$',
                        r'$\lambda_2$', r'$\lambda_2$',
                        r'$\lambda_5$', r'$\lambda_5$'))

    ax.set_xlim((-max(*map(abs, ax.get_xlim())),
                 max(*map(abs, ax.get_xlim()))))

    plt.savefig('./relative_vars.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (5.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (2.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (2.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for m in tqdm(np.linspace(-1, 1, 5).tolist(),
                  total=5,
                  desc='Iterate over the possible values',
                  file=sys.stdout,
                  colour='green',
                  mininterval=0.001,
                  maxinterval=0.01,
                  position=0,
                  leave=True):

        m_vals = (m, m, m)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m_vals,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 0], stress_path[:, 0], label=f"{m:.1f}")

    plt.xlabel(r'$H_{xx}$', fontsize=16)
    plt.ylabel(r'$\tau_{xx}$', fontsize=16)
    plt.legend(title=r'$m=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_1=5, \ \lambda_2=3, \ '
              r'\lambda_5=2, \ \sigma=1, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_m.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (5.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (2.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (2.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for sig in tqdm((0.1, 0.5, 1.0, 5.0),
                    total=4,
                    desc='Iterate over the possible values',
                    file=sys.stdout,
                    colour='green',
                    mininterval=0.001,
                    maxinterval=0.01,
                    position=0,
                    leave=True):

        sigma = (sig, 0.0, 0.0)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 0], stress_path[:, 0], label=sig)

    plt.xlabel(r'$H_{xx}$', fontsize=16)
    plt.ylabel(r'$\tau_{xx}$', fontsize=16)
    plt.legend(title=r'$\sigma=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_1=5, \ \lambda_2=3, \ '
              r'\lambda_5=2, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_sigma.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (5.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (3.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (2.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (2.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for angle in tqdm((0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2),
                      total=5,
                      desc='Iterate over the possible values',
                      file=sys.stdout,
                      colour='green',
                      mininterval=0.001,
                      maxinterval=0.01,
                      position=0,
                      leave=True):

        angles = (angle, 0.0, 0.0)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 0], stress_path[:, 0],
                 label=f"{round(np.rad2deg(angle), 0)}°")

    plt.xlabel(r'$H_{xx}$', fontsize=16)
    plt.ylabel(r'$\tau_{xx}$', fontsize=16)
    plt.legend(title=r'$\mu=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_1=5, \ \lambda_2=3, \ '
              r'\lambda_5=2, \ \sigma=1$', fontsize=16)
    plt.savefig('./figure_mu.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (1.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 1.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for lambda_5 in tqdm((10 ** (j - 1) for j in range(3)),
                         total=3,
                         desc='Iterate over the possible values',
                         file=sys.stdout,
                         colour='green',
                         mininterval=0.001,
                         maxinterval=0.01,
                         position=0,
                         leave=True):

        l4 = (lambda_5, 1.0, 1.0, 1.0, 1.0)
        l5 = (lambda_5, 1.0, 1.0, 1.0, 1.0)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 2], stress_path[:, 2], label=lambda_5)

    plt.xlabel(r'$H_{xy}$', fontsize=16)
    plt.ylabel(r'$\tau_{xy}$', fontsize=16)
    plt.legend(title=r'$\lambda_5=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_1=1, \ \lambda_2=1, \ '
              r'\sigma=1.0, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_lambda_5.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (1.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for lh in tqdm((10 ** j for j in range(4)),
                   total=4,
                   desc='Iterate over the possible values',
                   file=sys.stdout,
                   colour='green',
                   mininterval=0.001,
                   maxinterval=0.01,
                   position=0,
                   leave=True):

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(stress_path[:, 0], np.exp(strain_path[:, 0]
                                           + strain_path[:, 1]
                                           + ezz_path), label=lh)

    plt.xlabel(r'$\tau_{xx}$', fontsize=16)
    plt.ylabel('Volumetric change', fontsize=16)
    plt.legend(title=r'$\lambda_h=$')
    plt.title(r'$\lambda_1=1, \ \lambda_2=1, \ '
              r'\lambda_5=1, \ \sigma=1, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_lambda_h.svg', dpi=300)

    angles = (0.0, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (1.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for lambda_1 in tqdm((10 ** (j - 2) for j in range(4)),
                         total=4,
                         desc='Iterate over the possible values',
                         file=sys.stdout,
                         colour='green',
                         mininterval=0.001,
                         maxinterval=0.01,
                         position=0,
                         leave=True):

        l1 = (lambda_1, 1.0, 1.0, 1.0, 1.0)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 0], stress_path[:, 0], label=lambda_1)

    plt.xlabel(r'$H_{xx}$', fontsize=16)
    plt.ylabel(r'$\tau_{xx}$', fontsize=16)
    plt.legend(title=r'$\lambda_1=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_2=1, \ '
              r'\lambda_5=1, \ \sigma=1, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_lambda_1.svg', dpi=300)

    angles = (np.pi / 2, 0.0, 0.0)
    sigma = (1.0, 0.0, 0.0)
    dens = 1.0

    lh = 1000.0
    l1 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l2 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l3 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l4 = (1.0, 1.0, 1.0, 1.0, 1.0)
    l5 = (1.0, 1.0, 1.0, 1.0, 1.0)

    coeffs = (1.0, 0.0, 0.0, 0.0, 0.0)

    stress_path = np.stack((np.linspace(0.0, 1.0, 100),
                            np.linspace(0.0, 0.0, 100),
                            np.linspace(0.0, 0.0, 100)), axis=1)

    strain_path = np.zeros_like(stress_path)
    ezz_path = np.zeros_like(stress_path[:, 0])

    plt.figure()

    for lambda_2 in tqdm((10 ** (j - 1) for j in range(3)),
                         total=3,
                         desc='Iterate over the possible values',
                         file=sys.stdout,
                         colour='green',
                         mininterval=0.001,
                         maxinterval=0.01,
                         position=0,
                         leave=True):

        l2 = (lambda_2, 1.0, 1.0, 1.0, 1.0)
        l3 = (lambda_2, 1.0, 1.0, 1.0, 1.0)

        for i, stress in tqdm(enumerate(stress_path.tolist()),
                              total=stress_path.shape[0],
                              desc='Compute optimal strain tensor',
                              file=sys.stdout,
                              colour='green',
                              mininterval=0.001,
                              maxinterval=0.01,
                              position=1,
                              leave=False):
            stress = np.array(stress, dtype=np.float64)

            if (abs(stress[2]) < 0.001
                    and abs(stress[0] - stress[1]) < 0.001):
                m = (0.0, 0.0, 0.0)
            else:
                m = tuple(cos(2 * (theta - 0.5 * atan2(2 * stress[2],
                                                       stress[0] -
                                                       stress[1])))
                          for theta in angles)

            strain = least_squares(least_square_wrapper,
                                   x0=strain_path[i - 1] if i > 0
                                   else np.array((0.0, 0.0, 0.0),
                                                 dtype=np.float64),
                                   kwargs={'target_stress': stress,
                                           'm_vals': m,
                                           'lamh': lh,
                                           'lam1': l1,
                                           'lam2': l2,
                                           'lam3': l3,
                                           'lam4': l4,
                                           'lam5': l5,
                                           'vals': coeffs,
                                           'theta': angles,
                                           'sigstd': sigma,
                                           'density': dens}).x
            ezz_path[i] = EzzBuf.ezz
            strain_path[i] = strain

        plt.plot(strain_path[:, 0], stress_path[:, 0], label=lambda_2)

    plt.xlabel(r'$H_{xx}$', fontsize=16)
    plt.ylabel(r'$\tau_{xx}$', fontsize=16)
    plt.legend(title=r'$\lambda_2=$')
    plt.title(r'$\lambda_h=1000, \ \lambda_1=1, \ '
              r'\lambda_5=1, \ \sigma=1, \ \mu=0$', fontsize=16)
    plt.savefig('./figure_lambda_2.svg', dpi=300)
