# coding: utf-8

import cupy as cp
import math
from numba import cuda, types
import os

from .gpu_utils import (gpu_exp, gpu_pow, gpu_minus, gpu_sub, gpu_mul, gpu_div,
                        gpu_array_sub, gpu_array_mul, gpu_sum)

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.int32,
                           types.float32[:]),
          device=True)
def periodic_gauss_gpu(x: cp.ndarray,
                       sigma_1: float,
                       a_1: float,
                       sigma_2: float,
                       a_2: float,
                       sigma_3: float,
                       a_3: float,
                       b: float,
                       mu_1: float,
                       mu_2: float,
                       mu_3: float,
                       n: int,
                       array_out: cp.ndarray) -> cp.ndarray:
    """Computes the sum of three periodic gaussian curves of the input array,
    using the provided gaussian parameters.

    Args:
        x: The array over which to compute the gaussians.
        sigma_1: Standard deviation for the first gaussian.
        a_1: Multiplicative factor for the first gaussian.
        sigma_2: Standard deviation for the second gaussian.
        a_2: Multiplicative factor for the second gaussian.
        sigma_3: Standard deviation for the second gaussian.
        a_3: Multiplicative factor for the second gaussian.
        b: Offset, common to all the gaussians.
        mu_1: Center value of the first gaussian.
        mu_2: Center value of the second gaussian.
        mu_3: Center value of the third gaussian.
        n: Number of gaussians to compute, between 1 and 3.
        array_out: The array in which to store the result.

    Returns:
        The sum of the three periodic gaussians over the input array.
    """

    if n == 1:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)))
    elif n == 2:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)) +
                a_2 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_2,
                                                    math.pi) -
                                          math.pi / 2) / sigma_2, 2)))
    elif n == 3:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)) +
                a_2 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_2,
                                                    math.pi) -
                                          math.pi / 2) / sigma_2, 2)) +
                a_3 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_3,
                                                    math.pi) -
                                          math.pi / 2) / sigma_3, 2)))
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:],
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.float32,
                           types.int32,
                           types.float32[:]),
          device=True)
def periodic_gaussian_derivative(x: cp.ndarray,
                                 y: cp.ndarray,
                                 sigma_1: float,
                                 a_1: float,
                                 sigma_2: float,
                                 a_2: float,
                                 sigma_3: float,
                                 a_3: float,
                                 b: float,
                                 mu_1: float,
                                 mu_2: float,
                                 mu_3: float,
                                 n: int,
                                 array_out: cp.ndarray) -> cp.ndarray:
    """Computes the derivatives of the periodic gaussian defined previously
    with respect to all its parameters.

    Args:
        x: The array over which to compute the gaussian derivatives.
        y: Array containing the measured values from real-world data.
        sigma_1: Standard deviation for the first gaussian derivative.
        a_1: Multiplicative factor for the first gaussian derivative.
        sigma_2: Standard deviation for the second gaussian derivative.
        a_2: Multiplicative factor for the second gaussian derivative.
        sigma_3: Standard deviation for the second gaussian derivative.
        a_3: Multiplicative factor for the second gaussian derivative.
        b: Offset, common to all the gaussian derivatives.
        mu_1: Center value of the first gaussian derivative.
        mu_2: Center value of the second gaussian derivative.
        mu_3: Center value of the third gaussian derivative.
        n: Number of gaussian derivatives to compute, between 1 and 3.
        array_out: The array in which to store the result.

    Returns:
        An array containing for each of the parameters of the periodic gaussian
        the value of its derivative with respect to this parameter.
    """

    # Define buffers used during calculation and stored in memory
    buf = cuda.local.array((NB_ANGLES,), dtype=types.float32)
    diff = cuda.local.array((NB_ANGLES,), dtype=types.float32)
    exp_1 = cuda.local.array((NB_ANGLES,), dtype=types.float32)
    exp_2 = cuda.local.array((NB_ANGLES,), dtype=types.float32)
    exp_3 = cuda.local.array((NB_ANGLES,), dtype=types.float32)

    # The difference between the expected values and the model
    gpu_array_sub(y,
                  periodic_gauss_gpu(x,
                                     sigma_1,
                                     a_1,
                                     sigma_2,
                                     a_2,
                                     sigma_3,
                                     a_3,
                                     b,
                                     mu_1,
                                     mu_2,
                                     mu_3,
                                     n,
                                     buf),
                  diff)

    # exp_1 = np.exp(-((x - mu_1) / sigma_1) ** 2)
    gpu_exp(gpu_minus(gpu_pow(gpu_div(gpu_sub(x,
                                              mu_1,
                                              buf),
                                      sigma_1,
                                      buf),
                              2,
                              buf),
                      buf),
            exp_1)

    if n == 1:

        # -4 * a_1 * np.sum(diff * exp_1 * (x - mu_1) ** 2 / (sigma_1 ** 3))
        array_out[0] = (
            -4 * a_1 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_1,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_1,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_1 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_1)
        array_out[1] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_1,
                                                  buf))
        array_out[2] = 0.
        array_out[3] = 0.
        array_out[4] = 0.
        array_out[5] = 0.
        # -2 * np.sum(diff)
        array_out[6] = -2 * gpu_sum(diff)

    elif n == 2:

        # exp_2 = np.exp(-((x - mu_2) / sigma_2) ** 2)
        gpu_exp(gpu_minus(gpu_pow(gpu_div(gpu_sub(x,
                                                  mu_2,
                                                  buf),
                                          sigma_2,
                                          buf),
                                  2,
                                  buf),
                          buf),
                exp_2)

        # -4 * a_1 * np.sum(diff * exp_1 * (x - mu_1) ** 2 / (sigma_1 ** 3))
        array_out[0] = (
            -4 * a_1 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_1,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_1,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_1 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_1)
        array_out[1] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_1,
                                                  buf))
        # -4 * a_2 * np.sum(diff * exp_2 * (x - mu_2) ** 2 / (sigma_2 ** 3))
        array_out[2] = (
            -4 * a_2 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_2,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_2,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_2 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_2)
        array_out[3] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_2,
                                                  buf))
        # -2 * np.sum(diff)
        array_out[6] = -2 * gpu_sum(diff)

    elif n == 3:

        # exp_2 = np.exp(-((x - mu_2) / sigma_2) ** 2)
        gpu_exp(gpu_minus(gpu_pow(gpu_div(gpu_sub(x,
                                                  mu_2,
                                                  buf),
                                          sigma_2,
                                          buf),
                                  2,
                                  buf),
                          buf),
                exp_2)
        # exp_3 = np.exp(-((x - mu_3) / sigma_3) ** 2)
        gpu_exp(gpu_minus(gpu_pow(gpu_div(gpu_sub(x,
                                                  mu_3,
                                                  buf),
                                          sigma_3,
                                          buf),
                                  2,
                                  buf),
                          buf),
                exp_3)

        # -4 * a_1 * np.sum(diff * exp_1 * (x - mu_1) ** 2 / (sigma_1 ** 3))
        array_out[0] = (
            -4 * a_1 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_1,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_1,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_1 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_1)
        array_out[1] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_1,
                                                  buf))
        # -4 * a_2 * np.sum(diff * exp_2 * (x - mu_2) ** 2 / (sigma_2 ** 3))
        array_out[2] = (
            -4 * a_2 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_2,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_2,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_2 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_2)
        array_out[3] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_2,
                                                  buf))
        # -4 * a_3 * np.sum(diff * exp_3 * (x - mu_3) ** 2 / (sigma_3 ** 3))
        array_out[4] = (
            -4 * a_3 *
            gpu_sum(gpu_div(gpu_array_mul(diff,
                                          gpu_array_mul(exp_3,
                                                        gpu_pow(gpu_sub(x,
                                                                        mu_3,
                                                                        buf),
                                                                2,
                                                                buf),
                                                        buf),
                                          buf),
                            sigma_3 ** 3,
                            buf)))
        # -2 * np.sum(diff * exp_3)
        array_out[5] = -2 * gpu_sum(gpu_array_mul(diff,
                                                  exp_3,
                                                  buf))
        # -2 * np.sum(diff)
        array_out[6] = -2 * gpu_sum(diff)

    return array_out


@cuda.jit(types.float32[:](types.int32,
                           types.float32[:],
                           types.float32[:],
                           types.float32[:],
                           types.float32[:],
                           types.float32,
                           types.int32),
          device=True)
def gradient_descent(n_peak: int,
                     x_data: cp.ndarray,
                     y_data: cp.ndarray,
                     params: cp.ndarray,
                     mu: cp.ndarray,
                     thresh: float,
                     max_iter: int) -> cp.ndarray:
    """Performs gradient descent to find the set of parameters that minimizes
    the difference between the computed periodic gaussians and the response to
    the Gabor filters.

    Args:
        n_peak: Number of peaks over which to compute the periodic gaussian.
        x_data: Array containing the angles over which to compute the
            periodic gaussian.
        y_data: Array containing measured data, target to the gradient descent.
        params: Array containing a first estimation of the optimal parameters
            of the periodic gaussian.
        mu: Array containing the angular location of the centers of the
            gaussians.
        thresh: Once the difference between two residuals is lower than this
            value, the algorithm stops.
        max_iter: Once that many iterations have been executed, the algorithm
            stops.

    Returns:
        The set of parameters that minimize the difference between the computed
        data and the periodic gaussians.
    """

    if n_peak == 0:
        return params

    buf = cuda.local.array((NB_ANGLES,), dtype=types.float32)
    buf_7 = cuda.local.array((7,), dtype=types.float32)

    weight = 0.001
    residuals = gpu_sum(gpu_pow(gpu_array_sub(y_data,
                                              periodic_gauss_gpu(x_data,
                                                                 params[0],
                                                                 params[1],
                                                                 params[2],
                                                                 params[3],
                                                                 params[4],
                                                                 params[5],
                                                                 params[6],
                                                                 mu[0],
                                                                 mu[1],
                                                                 mu[2],
                                                                 n_peak,
                                                                 buf),
                                              buf),
                                2,
                                buf))
    gpu_array_sub(params,
                  gpu_mul(periodic_gaussian_derivative(x_data,
                                                       y_data,
                                                       params[0],
                                                       params[1],
                                                       params[2],
                                                       params[3],
                                                       params[4],
                                                       params[5],
                                                       params[6],
                                                       mu[0],
                                                       mu[1],
                                                       mu[2],
                                                       n_peak,
                                                       buf_7),
                          weight,
                          buf_7),
                  params)
    new_res = gpu_sum(gpu_pow(gpu_array_sub(y_data,
                                            periodic_gauss_gpu(x_data,
                                                               params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               params[5],
                                                               params[6],
                                                               mu[0],
                                                               mu[1],
                                                               mu[2],
                                                               n_peak,
                                                               buf),
                                            buf),
                              2,
                              buf))

    n = 0
    while True:
        if ((new_res < residuals and residuals - new_res < thresh)
            or n > max_iter):
            break
        n += 1

        weight = 1.2 * weight if new_res < residuals else 0.5 * weight

        gpu_array_sub(params,
                      gpu_mul(periodic_gaussian_derivative(x_data,
                                                           y_data,
                                                           params[0],
                                                           params[1],
                                                           params[2],
                                                           params[3],
                                                           params[4],
                                                           params[5],
                                                           params[6],
                                                           mu[0],
                                                           mu[1],
                                                           mu[2],
                                                           n_peak,
                                                           buf_7),
                              weight,
                              buf_7),
                      params)
        residuals = new_res
        new_res = gpu_sum(gpu_pow(gpu_array_sub(y_data,
                                                periodic_gauss_gpu(x_data,
                                                                   params[0],
                                                                   params[1],
                                                                   params[2],
                                                                   params[3],
                                                                   params[4],
                                                                   params[5],
                                                                   params[6],
                                                                   mu[0],
                                                                   mu[1],
                                                                   mu[2],
                                                                   n_peak,
                                                                   buf),
                                                buf),
                                  2,
                                  buf))

    return params
