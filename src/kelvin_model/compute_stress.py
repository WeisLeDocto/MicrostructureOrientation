# coding: utf-8

import ctypes
import numpy as np
from pathlib import Path


def compute_stress(lib_path: Path,
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
                   density: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the stress on all the specified pixels using a dedicated C++ 
    library.

    Args:
        lib_path: Path to the .so file containing the shared library for
          computing the stress.
        exx: Numpy array containing for all pixels the xx strain.
        eyy: Numpy array containing for all pixels the yy strain.
        exy: Numpy array containing for all pixels the xy strain.
        lambda_h: Value of the hydrostatic eigenvalue, common to all orders.
        lambda_11: Value of the first deviatoric eigenvalue for order 1.
        lambda_21: Value of the second deviatoric eigenvalue for order 1.
        lambda_51: Value of the fifth deviatoric eigenvalue for order 1.
        lambda_12: Value of the first deviatoric eigenvalue for order 2.
        lambda_22: Value of the second deviatoric eigenvalue for order 2.
        lambda_52: Value of the fifth deviatoric eigenvalue for order 2.
        lambda_13: Value of the first deviatoric eigenvalue for order 3.
        lambda_23: Value of the second deviatoric eigenvalue for order 3.
        lambda_53: Value of the fifth deviatoric eigenvalue for order 3.
        lambda_14: Value of the first deviatoric eigenvalue for order 4.
        lambda_24: Value of the second deviatoric eigenvalue for order 4.
        lambda_54: Value of the fifth deviatoric eigenvalue for order 4.
        lambda_15: Value of the first deviatoric eigenvalue for order 5.
        lambda_25: Value of the second deviatoric eigenvalue for order 5.
        lambda_55: Value of the fifth deviatoric eigenvalue for order 5.
        val1: Multiplicative factor for the first-order term.
        val2: Multiplicative factor for the second-order term.
        val3: Multiplicative factor for the third-order term.
        val4: Multiplicative factor for the fourth-order term.
        val5: Multiplicative factor for the fifth-order term.
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
        Three numpy arrays containing respectively the xx, yy, and xy stress
        for all pixels.
    """

    # Load the shared C++ library
    lib = ctypes.CDLL(str(lib_path))
    
    # Arrays for storing the result stress
    sxx = np.empty_like(exx, dtype=np.float64)
    syy = np.empty_like(exx, dtype=np.float64)
    sxy = np.empty_like(exx, dtype=np.float64)
    
    # The arrays must be contiguous before being passed to C++ code
    sxx = np.ascontiguousarray(sxx)
    syy = np.ascontiguousarray(syy)
    sxy = np.ascontiguousarray(sxy)
    theta_1 = np.ascontiguousarray(theta_1)
    theta_2 = np.ascontiguousarray(theta_2)
    theta_3 = np.ascontiguousarray(theta_3)
    sigma_1 = np.ascontiguousarray(sigma_1)
    sigma_2 = np.ascontiguousarray(sigma_2)
    sigma_3 = np.ascontiguousarray(sigma_3)
    density = np.ascontiguousarray(density)
    exx = np.ascontiguousarray(exx)
    eyy = np.ascontiguousarray(eyy)
    exy = np.ascontiguousarray(exy)
    
    # Compute the stress only on the diagonals
    lib.calc_stresses(
        exx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        eyy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        exy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        *map(ctypes.c_double, (lambda_h,
                               lambda_11,
                               lambda_21,
                               lambda_51,
                               lambda_51,
                               lambda_51,
                               lambda_12,
                               lambda_22,
                               lambda_51,
                               lambda_51,
                               lambda_52,
                               lambda_13,
                               lambda_23,
                               lambda_51,
                               lambda_51,
                               lambda_53,
                               lambda_14,
                               lambda_24,
                               lambda_51,
                               lambda_51,
                               lambda_54,
                               lambda_15,
                               lambda_25,
                               lambda_51,
                               lambda_51,
                               lambda_55,
                               val1,
                               val2,
                               val3,
                               val4,
                               val5)),
        theta_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta_3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma_3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(exx.shape[0]),
        ctypes.c_int(exx.shape[1]),
        sxx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        syy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sxy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    
    return sxx, syy, sxy
