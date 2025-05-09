# coding: utf-8

from pathlib import Path
import cupy as cp
import numpy as np
from tqdm.auto import tqdm
import math
import sys
from numba import cuda, types
import os

from .gpu_utils import gpu_array_copy
from .periodic_gaussian import gradient_descent

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


@cuda.jit(types.void(types.int32[:, :],
                     types.float32[:],
                     types.float32[:, :, :],
                     types.float32[:, :, :],
                     types.float32[:, :, :],
                     types.float32,
                     types.int32))
def _fit_gpu(n_peak: cp.ndarray,
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
        _fit_gpu[bpg, tpb](n_gpu, x_gpu, y_gpu, p_gpu, m_gpu, 1e-6, 5000)

        # Copy the result in CPU memory and write it on the disk
        param = p_gpu.copy_to_host()
        np.save(dest_path / f'{img_name}.npy', param)

        # Free up the GPU memory
        mem_pool.free_all_blocks()
