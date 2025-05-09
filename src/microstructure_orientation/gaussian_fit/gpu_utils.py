# coding:utf-8

import cupy as cp
import math
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
