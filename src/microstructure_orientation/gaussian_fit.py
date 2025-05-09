# coding: utf-8

from pathlib import Path
import cupy as cp
import numpy as np
from tqdm.auto import tqdm
import math
import sys
from numba import cuda, types
import os

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_exp(array_in: cp.ndarray,
            array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise exponentiation on a 1-dimensional array."""

    for i in range(array_in.shape[0]):
        array_out[i] = math.exp(array_in[i])
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.int32,
                           types.float32[:]),
          device=True)
def gpu_pow(array_in: cp.ndarray,
            power: int,
            array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise elevation to the given power on a 1-dimensional
    array."""

    for i in range(array_in.shape[0]):
        array_out[i] = math.pow(array_in[i], power)
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_minus(array_in: cp.ndarray,
              array_out: cp.ndarray) -> cp.ndarray:
    """Equivalent to multiplying a 1-dimensional array by -1."""

    for i in range(array_in.shape[0]):
       array_out[i] = -array_in[i]
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32,
                           types.float32[:]),
          device=True)
def gpu_sub(array_in: cp.ndarray,
            val: float,
            array_out: cp.ndarray) -> cp.ndarray:
    """Subtracts a constant value to all the elements of a 1-dimensional
    array."""

    for i in range(array_in.shape[0]):
        array_out[i] = array_in[i] - val
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32,
                           types.float32[:]),
          device=True)
def gpu_mul(array_in: cp.ndarray,
            val: float,
            array_out: cp.ndarray) -> cp.ndarray:
    """Multiplies all the elements of a 1-dimensional array by a constant
    value."""

    for i in range(array_in.shape[0]):
        array_out[i] = array_in[i] * val
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32,
                           types.float32[:]),
          device=True)
def gpu_div(array_in: cp.ndarray,
            val: float,
            array_out: cp.ndarray) -> cp.ndarray:
    """Divides all the elements of a 1-dimensional array by a constant
    value."""

    for i in range(array_in.shape[0]):
        array_out[i] = array_in[i] / val
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_array_add(array_in_1: cp.ndarray,
                  array_in_2: cp.ndarray,
                  array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise addition of two 1-dimensional arrays."""

    for i in range(array_in_1.shape[0]):
        array_out[i] = array_in_1[i] + array_in_2[i]
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_array_sub(array_in_1: cp.ndarray,
                  array_in_2: cp.ndarray,
                  array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise subtraction of two 1-dimensional arrays."""

    for i in range(array_in_1.shape[0]):
        array_out[i] = array_in_1[i] - array_in_2[i]
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_array_mul(array_in_1: cp.ndarray,
                  array_in_2: cp.ndarray,
                  array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise multiplication of two 1-dimensional arrays."""

    for i in range(array_in_1.shape[0]):
        array_out[i] = array_in_1[i] * array_in_2[i]
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_array_div(array_in_1: cp.ndarray,
                  array_in_2: cp.ndarray,
                  array_out: cp.ndarray) -> cp.ndarray:
    """Performs element-wise division of two 1-dimensional arrays."""

    for i in range(array_in_1.shape[0]):
        array_out[i] = array_in_1[i] / array_in_2[i]
    return array_out


@cuda.jit(types.float32[:](types.float32[:],
                           types.float32[:]),
          device=True)
def gpu_array_copy(array_in_1: cp.ndarray,
                   array_in_2: cp.ndarray) -> cp.ndarray:
    """Copies the values of a 1-dimensional array into another array."""

    for i in range(array_in_1.shape[0]):
        array_in_2[i] = array_in_1[i]
    return array_in_2


@cuda.jit(types.float32(types.float32[:]),
          device=True)
def gpu_sum(array_in: cp.ndarray) -> float:
    """Computes the sum of all the elements of a 1-dimensional array."""

    ret = 0.
    for i in range(array_in.shape[0]):
        ret += array_in[i]
    return ret


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


@cuda.jit(types.void(types.int32[:, :],
                     types.float32[:],
                     types.float32[:, :, :],
                     types.float32[:, :, :],
                     types.float32[:, :, :],
                     types.float32,
                     types.int32))
def fit_gpu(n_peak: cp.ndarray,
            x_data: cp.ndarray,
            y_data: cp.ndarray,
            params: cp.ndarray,
            mu: cp.ndarray,
            thresh: float,
            max_iter: int) -> None:
    """Wrapper around the gradient_descent function, to apply it on all the
    pixels of the input image.

    Args:
        n_peak: Number of detected peaks in the Gabor response.
        x_data: Array containing the angles over which the Gabor response was
            computed.
        y_data: Array containing the angular response to Gabor filters.
        params: Array containing a first estimation of the optimal parameters
            of the periodic gaussian.
        mu: Array containing the angular location of the centers of the
            gaussians.
        thresh: Once the difference between two residuals is lower than this
            value, the algorithm stops.
        max_iter: Once that many iterations have been executed, the algorithm
            stops.
    """

    x, y = cuda.grid(2)
    # if x == 1 and y == 3:
    #   from pdb import set_trace; set_trace()
    if x < params.shape[0] and y < params.shape[1]:
        buf_7 = cuda.local.array((7,), dtype=types.float32)
        gpu_array_copy(params[x, y], buf_7)
        gradient_descent(n_peak[x, y],
                         x_data,
                         y_data[x, y],
                         buf_7,
                         mu[x, y],
                         thresh,
                         max_iter)
        gpu_array_copy(buf_7, params[x, y])


def gaussian_fit(peak_folder: Path,
                 gabor_folder: Path,
                 dest_path: Path) -> None:
    """From the response to Gabor filter, and a set of estimated parameters,
    determines for each pixel the set of parameters that give the best fit of
    the response with periodic gaussian curves.

    Args:
        peak_folder: Path to the folder containing the sets of estimated
            parameters for the gaussian fit.
        gabor_folder: Path to the folder containing the responses to the Gabor
            filter.
        dest_path: Path to the folder where to write the set of optimized
            parameters.
    """

    # Create the destination folder and load the input data
    dest_path.mkdir(parents=False, exist_ok=True)
    peak_paths = tuple(path.stem for path in peak_folder.glob('*.npz'))

    # Memory manager to be able to free up the memory
    mem_pool = cp.get_default_memory_pool()

    # Fit a gaussian curve on the response to Gabor filter for each image
    for img_name in tqdm(peak_paths,
                         total=len(peak_paths),
                         desc='Fitting Gaussian curves',
                         file=sys.stdout,
                         colour='green'):
        data = np.load(peak_folder / f'{img_name}.npz')
        peaks, param = data['arr_0'], data['arr_1']
        res = np.load(gabor_folder / f'{img_name}.npy')

        # These parameters should work well on most GPU
        tpb = (16, 16)
        bpg = (int(math.ceil(res.shape[0] / tpb[0])),
               int(math.ceil(res.shape[1] / tpb[1])))

        # Count the number of peaks and load in GPU memory
        n_peaks = np.count_nonzero(np.invert(np.isnan(peaks)), axis=-1)
        n_gpu = cuda.to_device(n_peaks.astype(np.float32))

        # Load all the data in GPU memory
        x_gpu = cuda.to_device(
            np.radians(np.linspace(0, 180, NB_ANGLES)).astype(np.float32))
        y_gpu = cuda.to_device(res.astype(np.float32))
        p_gpu = cuda.to_device(param.astype(np.float32))
        m_gpu = cuda.to_device(np.radians(peaks).astype(np.float32))

        # Perform the gaussian fit
        fit_gpu[bpg, tpb](n_gpu, x_gpu, y_gpu, p_gpu, m_gpu, 1e-6, 5000)

        # Copy the result in CPU memory and write it on the disk
        param = p_gpu.copy_to_host()
        np.save(dest_path / f'{img_name}.npy', param)

        # Free up the GPU memory
        mem_pool.free_all_blocks()
