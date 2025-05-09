# coding: utf-8

import cupy as cp
from numba import cuda, types
import os

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


@cuda.jit(types.void(types.float32[:, :, :],
                     types.int32[:, :],
                     types.float32[:, :, :]))
def rearrange(array_in: cp.ndarray,
              min_idx: cp.ndarray,
              array_out: cp.ndarray) -> None:
    """Rearranges the elements of 1-dimensional arrays on a 3D grid so that the
    minimum value is located on one end of the new array.

    Args:
        array_in: The arrays to rearrange.
        min_idx: The indexes of the minimum values.
        array_out: Buffer for the rearranged arrays.
    """

    x, y = cuda.grid(2)
    if x < array_in.shape[0] and y < array_in.shape[1]:
        dim = array_in.shape[2]
        idx = min_idx[x, y]
        for i in range(dim - idx):
            array_out[x, y, i] = array_in[x, y, idx + i]
        for i in range(idx):
            array_out[x, y, dim - idx + i] = array_in[x, y, i]


@cuda.jit(types.void(types.int32[:, :, :],
                     types.int32[:, :, :]))
def make_peak_mask(peaks: cp.ndarray,
                   mask: cp.ndarray) -> None:
    """Generates a mask of 0 and 1 indicating for each point on a 3D grid if it
    is the location of a peak or not.

    Args:
        peaks: Array containing the indexes of the peaks.
        mask: Array containing the peak mask.
    """

    x, y, z = cuda.grid(3)
    if x < peaks.shape[0] and y < peaks.shape[1] and z < peaks.shape[2]:
        peak_idx = peaks[x, y, z]
        if peak_idx > 0:
            mask[x, y, peak_idx] = 1


@cuda.jit(types.void(types.float32[:, :, :],
                     types.int32[:, :, :]))
def local_maxima_gpu(data: cp.ndarray,
                     midpoints: cp.ndarray) -> None:
    """Finds the local maxima of 1D arrays on a 3D grid.

    Args:
        data: Raw data over which to detect the local maxima.
        midpoints: Array containing either -1 where there is no local maximum,
            or the index of the local maximum.
    """

    x, y, z = cuda.grid(3)
    if x < data.shape[0] and y < data.shape[1] and z < data.shape[2] - 2:

        z_max = data.shape[2] - 1
        midpoint = -1

        if data[x, y, z] < data[x, y, z + 1]:
            ahead = z + 2

            while (ahead < z_max and
                   data[x, y, ahead] == data[x, y, z + 1]):
                ahead += 1

            if data[x, y, ahead] < data[x, y, z + 1]:
                midpoint = (z + ahead) // 2

        midpoints[x, y, z] = midpoint


@cuda.jit(types.void(types.float32[:, :, :],
                     types.int32[:, :, :],
                     types.float32[:, :, :],
                     types.int32[:, :, :],
                     types.int32[:, :, :]))
def peak_prominences_gpu(data: cp.ndarray,
                         peak_mask: cp.ndarray,
                         prominences: cp.ndarray,
                         left_bases: cp.ndarray,
                         right_bases: cp.ndarray) -> None:
    """Computes the prominence of the detected peaks, as well as the extent of
    their left and right bases.

    Args:
        data: The raw data over which to compute the peaks prominences.
        peak_mask: A mask indicating where the peaks are located.
        prominences: The array where the prominences values will be stored.
        left_bases: Array for storing the extent of the left base of the peaks.
        right_bases: Array for storing the extent of the right base of the
            peaks.
    """

    x, y, z = cuda.grid(3)
    if x < data.shape[0] and y < data.shape[1] and z < data.shape[2]:
        if peak_mask[x, y, z] > 0:

            i_min = 0
            i_max = data.shape[2] - 1

            left_bases[x, y, z] = z
            i = z
            left_min = data[x, y, z]

            while i_min <= i and data[x, y, i] <= data[x, y, z]:
                if data[x, y, i] < left_min:
                    left_min = data[x, y, i]
                    left_bases[x, y, z] = i
                i -= 1

            right_bases[x, y, z] = z
            i = z
            right_min = data[x, y, z]

            while i <= i_max and data[x, y, i] <= data[x, y, z]:
                if data[x, y, i] < right_min:
                    right_min = data[x, y, i]
                    right_bases[x, y, z] = i
                i += 1

            prominences[x, y, z] = data[x, y, z] - max(left_min, right_min)


@cuda.jit(types.void(types.float32[:, :, :],
                     types.int32[:, :, :],
                     types.float32[:, :, :],
                     types.int32[:, :, :],
                     types.int32[:, :, :],
                     types.float32[:, :, :],
                     types.float32[:, :, :]))
def peak_width_gpu(data: cp.ndarray,
                   peak_mask: cp.ndarray,
                   prominences: cp.ndarray,
                   left_bases: cp.ndarray,
                   right_bases: cp.ndarray,
                   widths: cp.ndarray,
                   width_heights: cp.ndarray) -> None:
    """Computes the width of the detected peaks, as well as the height at which
    this width was calculated.

    Args:
        data: The raw data over which to compute the peaks widths.
        peak_mask: A mask indicating where the peaks are located.
        prominences: Array containing the prominences of the peaks.
        left_bases: Array containing the extent of the left bases of the peaks.
        right_bases: Array containing the extent of the right bases of the
            peaks.
        widths: Array where to store the widths of the peaks.
        width_heights: Array where to store the height at which the widths of
            the arrays are located.
    """

    x, y, z = cuda.grid(3)
    if x < data.shape[0] and y < data.shape[1] and z < data.shape[2]:
        if peak_mask[x, y, z] > 0:
            i_min = left_bases[x, y, z]
            i_max = right_bases[x, y, z]

            height = data[x, y, z] - prominences[x, y, z] * 0.5
            width_heights[x, y, z] = height

            i = z
            while i_min < i and height < data[x, y, i]:
                i -= 1

            left_ip = i
            if data[x, y, i] < height:
                left_ip += ((height - data[x, y, i]) /
                            (data[x, y, i + 1] - data[x, y, i]))

            i = z
            while i < i_max and height < data[x, y, i]:
                i += 1

            right_ip = i
            if data[x, y, i] < height:
                right_ip -= ((height - data[x, y, i]) /
                             (data[x, y, i - 1] - data[x, y, i]))

            widths[x, y, z] = right_ip - left_ip
