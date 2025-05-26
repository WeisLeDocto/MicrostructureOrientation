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
    
    # Array for storing the result stress
    stress = np.zeros((*exx.shape, 3), dtype=np.float64, order='C')

    # Store the
    input_dtype = np.dtype([('exx', np.float64), ('eyy', np.float64),
                            ('exy', np.float64), ('theta_1', np.float64),
                            ('theta_2', np.float64), ('theta_3', np.float64),
                            ('sigma_1', np.float64), ('sigma_2', np.float64),
                            ('sigma_3', np.float64), ('density', np.float64)],
                           align=True)
    input_data = np.zeros(exx.shape, dtype=input_dtype)
    input_data['exx'] = exx
    input_data['eyy'] = eyy
    input_data['exy'] = exy
    input_data['theta_1'] = theta_1
    input_data['theta_2'] = theta_2
    input_data['theta_3'] = theta_3
    input_data['sigma_1'] = sigma_1
    input_data['sigma_2'] = sigma_2
    input_data['sigma_3'] = sigma_3
    input_data['density'] = density

    lambda_dtype = np.dtype([('lambda_h', np.float64),
                             ('lambda_11', np.float64),
                             ('lambda_21', np.float64),
                             ('lambda_31', np.float64),
                             ('lambda_41', np.float64),
                             ('lambda_51', np.float64),
                             ('lambda_12', np.float64),
                             ('lambda_22', np.float64),
                             ('lambda_32', np.float64),
                             ('lambda_42', np.float64),
                             ('lambda_52', np.float64),
                             ('lambda_13', np.float64),
                             ('lambda_23', np.float64),
                             ('lambda_33', np.float64),
                             ('lambda_43', np.float64),
                             ('lambda_53', np.float64),
                             ('lambda_14', np.float64),
                             ('lambda_24', np.float64),
                             ('lambda_34', np.float64),
                             ('lambda_44', np.float64),
                             ('lambda_54', np.float64),
                             ('lambda_15', np.float64),
                             ('lambda_25', np.float64),
                             ('lambda_35', np.float64),
                             ('lambda_45', np.float64),
                             ('lambda_55', np.float64),
                             ('val1', np.float64),
                             ('val2', np.float64),
                             ('val3', np.float64),
                             ('val4', np.float64),
                             ('val5', np.float64)], align=True)
    lambda_params = np.array((lambda_h, lambda_11, lambda_21, lambda_51,
                              lambda_51, lambda_51, lambda_12, lambda_22,
                              lambda_52, lambda_52, lambda_52, lambda_13,
                              lambda_23, lambda_53, lambda_53, lambda_53,
                              lambda_14, lambda_24, lambda_54, lambda_54,
                              lambda_54, lambda_15, lambda_25, lambda_55,
                              lambda_55, lambda_55, val1, val2, val3, val4,
                              val5),
                             dtype=lambda_dtype)
    
    # The arrays must be contiguous before being passed to C++ code
    stress = np.ascontiguousarray(stress)
    input_data = np.ascontiguousarray(input_data)
    lambda_params = np.ascontiguousarray(lambda_params)
    
    # Compute the stress only on the diagonals
    lib.calc_stresses(
        input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lambda_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(exx.shape[0]),
        ctypes.c_int(exx.shape[1]),
        stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    return stress[..., 0], stress[..., 1], stress[..., 2]
