# coding: utf-8

from pathlib import Path
import cupy as cp
import numpy as np
from tqdm.auto import tqdm
import sys
import math
import os

from .peak_utils import (rearrange, make_peak_mask, local_maxima_gpu,
                         peak_prominences_gpu, peak_width_gpu)

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def _find_peaks_gpu(gabor_data_cpu: np.ndarray,
                    ang_cpu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given the angular response to Gabor filters, detects up to three peaks
    with sufficient prominence and width in the signal. Also computes an
    estimation of parameters to fit a periodic gaussian curve on the response.

    Args:
        gabor_data_cpu: Angular response to the Gabor filter.
        ang_cpu: Array of angles over which the response to the Gabor filter
            was computed.

    Returns:
        The location of the peaks, and an estimation of parameters for fitting
        a gaussian curve over the angular response.
    """

    # Load data into GPU memory
    gabor_data = cp.asarray(gabor_data_cpu, dtype=cp.float32)
    ang = cp.asarray(ang_cpu, dtype=cp.float32)
    del gabor_data_cpu

    # These parameters should work well on most GPU
    tpb = (8, 8, 8)
    bpg = (int(math.ceil(gabor_data.shape[0] / tpb[0])),
           int(math.ceil(gabor_data.shape[1] / tpb[1])),
           int(math.ceil(gabor_data.shape[2] / tpb[2])))

    # Rearrange the Gabor response curve so that the minimum is on one end of
    # the array
    min_idx = cp.argmin(gabor_data, axis=2, dtype=cp.int32)
    to_search = cp.empty_like(gabor_data, dtype=cp.float32)
    rearrange[bpg, tpb](gabor_data, min_idx, to_search)

    # Find the indexes of the local maxima
    peaks = cp.empty_like(gabor_data, dtype=cp.int32)
    local_maxima_gpu[bpg, tpb](to_search, peaks)

    # Make a mask indicating for each point if it is a local maximum
    peak_mask = cp.zeros_like(gabor_data, dtype=cp.int32)
    make_peak_mask[bpg, tpb](peaks, peak_mask)
    del peaks

    # Compute the prominence of each peak
    prominences = cp.zeros_like(gabor_data, dtype=cp.float32)
    left_bases = cp.zeros_like(gabor_data, dtype=cp.int32)
    right_bases = cp.zeros_like(gabor_data, dtype=cp.int32)
    peak_prominences_gpu[bpg, tpb](to_search, peak_mask, prominences,
                                   left_bases, right_bases)

    # Set a threshold at 5% of total amplitude, and eliminate local maxima
    # whose prominence is below threshold
    min_val = cp.min(gabor_data, axis=2)
    min_prominence = 0.05 * (cp.max(gabor_data, axis=2) - min_val)
    prominence_mask = prominences < min_prominence[:, :, cp.newaxis]
    peak_mask[prominence_mask] = 0
    prominences[prominence_mask] = 0
    del min_prominence, prominence_mask

    # Extract the width of each remaining peak
    widths = cp.zeros_like(gabor_data, dtype=cp.float32)
    width_heights = cp.zeros_like(gabor_data, dtype=cp.float32)
    peak_width_gpu[bpg, tpb](to_search, peak_mask, prominences, left_bases,
                             right_bases, widths, width_heights)
    del left_bases, right_bases

    # Compute the height of each peak, plus auxiliary values for later
    heights = to_search - min_val[:, :, cp.newaxis]
    widths *= cp.radians(ang[1] - ang[0])
    width_heights -= min_val[:, :, cp.newaxis]

    # Sort the remaining peaks by prominence
    sorted_order = cp.argsort(prominences, axis=2)[:, :, ::-1]
    del prominences

    # Only keep up to 3 peaks among the ones that were remaining, and reorder
    # the arrays back to the original order
    heights_final = cp.take_along_axis(heights, sorted_order, axis=2)[:, :, :3]
    heights_final[heights_final <= 0] = 1
    del heights
    widths_final = cp.take_along_axis(widths, sorted_order, axis=2)[:, :, :3]
    widths_final[widths_final <= 0] = 1
    del widths
    width_heights_final = cp.take_along_axis(width_heights, sorted_order,
                                             axis=2)[:, :, :3]
    width_heights_final[width_heights_final <= 0] = 1
    del width_heights
    peak_mask_final = cp.take_along_axis(peak_mask, sorted_order,
                                         axis=2)[:, :, :3]
    del peak_mask
    peak_index = sorted_order[:, :, :3]
    peak_index_final = (peak_index +
                        min_idx[:, :, cp.newaxis]) % to_search.shape[2]
    del peak_index, min_idx, to_search, sorted_order

    # Compute an approximation of a standard deviation
    deviation_final = widths_final / (2 * cp.sqrt(cp.log(heights_final /
                                                         width_heights_final)))
    del width_heights_final, widths_final

    # Convert the indexes of the peaks to actual angles
    peak_value_final = ang[peak_index_final]
    del peak_index_final

    # Fill output arrays with nan where no peak could be detected
    invalid_peak_mask_final = peak_mask_final <= 0
    del peak_mask_final
    deviation_final[invalid_peak_mask_final] = cp.nan
    heights_final[invalid_peak_mask_final] = cp.nan
    peak_value_final[invalid_peak_mask_final] = cp.nan
    del invalid_peak_mask_final

    # Populate the output array with the computed values
    params = cp.full((*gabor_data.shape[:2], 7), -1, dtype=cp.float32)
    params[:, :, -1] = min_val
    del min_val
    params[:, :, 0:6:2] = deviation_final
    del deviation_final
    params[:, :, 1:6:2] = heights_final
    del heights_final

    # Return numpy arrays to store on the disk
    return peak_value_final.get(), params.get()


def detect_peaks(src_path: Path,
                 dest_path: Path) -> None:
    """Given the angular response to Gabor filters, detects up to three peaks
    with sufficient prominence and width in the signal. Also computes an
    estimation of parameters to fit a periodic gaussian curve on the response.

    Args:
        src_path: Path to the folder containing the angular response to Gabor
            filter.
        dest_path: Path to the folder where to store the detected peaks.
    """

    # Create the destination folder and load the input data
    dest_path.mkdir(parents=False, exist_ok=True)
    gabor_data = tuple(sorted(src_path.glob('*.npy')))

    # Memory manager to be able to free up the memory
    mem_pool = cp.get_default_memory_pool()

    # Determine the location of the peaks and the parameters for each image
    for path in tqdm(gabor_data,
                     total=len(gabor_data),
                     desc='Detecting peaks',
                     file=sys.stdout,
                     colour='green'):
      peaks, params = _find_peaks_gpu(np.load(path),
                                      np.linspace(0, 180, NB_ANGLES))

      # Save the peaks location and their parameters at the indicated location
      np.savez(dest_path / f'{path.stem}.npz', peaks, params)

      # Free up the memory
      mem_pool.free_all_blocks()
