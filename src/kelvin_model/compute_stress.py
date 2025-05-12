# coding: utf-8

import ctypes
import concurrent.futures
import numpy as np
import itertools
from pathlib import Path
from tqdm.auto import tqdm
import sys
from math import prod


def _worker(lib_path: Path,
            i: int,
            exx: float,
            eyy: float,
            exy: float,
            lambda_h: float,
            lambda_11: float,
            lambda_21: float,
            lambda_31: float,
            lambda_41: float,
            lambda_51: float,
            lambda_12: float,
            lambda_22: float,
            lambda_32: float,
            lambda_42: float,
            lambda_52: float,
            lambda_13: float,
            lambda_23: float,
            lambda_33: float,
            lambda_43: float,
            lambda_53: float,
            lambda_14: float,
            lambda_24: float,
            lambda_34: float,
            lambda_44: float,
            lambda_54: float,
            lambda_15: float,
            lambda_25: float,
            lambda_35: float,
            lambda_45: float,
            lambda_55: float,
            val1: float,
            val2: float,
            val3: float,
            val4: float,
            val5: float,
            theta_1: float,
            theta_2: float,
            theta_3: float,
            sigma_1: float,
            sigma_2: float,
            sigma_3: float,
            density: float,
            sxx: ctypes.c_double,
            syy: ctypes.c_double,
            sxy: ctypes.c_double) -> tuple[int, float, float, float]:
    """Wrapper around the C++ executable that performs the stress computation.

    Args:
        lib_path: Path to the .so file containing the shared library for
          computing the stress.
        i: Index of the pixel on which to perform the calculation.
        exx: Value of the xx strain.
        eyy: Value of the yy strain.
        exy: Value of the xy strain.
        lambda_h: Value of the hydrostatic eigenvalue, common to all orders.
        lambda_11: Value of the first deviatoric eigenvalue for order 1.
        lambda_21: Value of the second deviatoric eigenvalue for order 1.
        lambda_31: Value of the third deviatoric eigenvalue for order 1.
        lambda_41: Value of the fourth deviatoric eigenvalue for order 1.
        lambda_51: Value of the fifth deviatoric eigenvalue for order 1.
        lambda_12: Value of the first deviatoric eigenvalue for order 2.
        lambda_22: Value of the second deviatoric eigenvalue for order 2.
        lambda_32: Value of the third deviatoric eigenvalue for order 2.
        lambda_42: Value of the fourth deviatoric eigenvalue for order 2.
        lambda_52: Value of the fifth deviatoric eigenvalue for order 2.
        lambda_13: Value of the first deviatoric eigenvalue for order 3.
        lambda_23: Value of the second deviatoric eigenvalue for order 3.
        lambda_33: Value of the third deviatoric eigenvalue for order 3.
        lambda_43: Value of the fourth deviatoric eigenvalue for order 3.
        lambda_53: Value of the fifth deviatoric eigenvalue for order 3.
        lambda_14: Value of the first deviatoric eigenvalue for order 4.
        lambda_24: Value of the second deviatoric eigenvalue for order 4.
        lambda_34: Value of the third deviatoric eigenvalue for order 4.
        lambda_44: Value of the fourth deviatoric eigenvalue for order 4.
        lambda_54: Value of the fifth deviatoric eigenvalue for order 4.
        lambda_15: Value of the first deviatoric eigenvalue for order 5.
        lambda_25: Value of the second deviatoric eigenvalue for order 5.
        lambda_35: Value of the third deviatoric eigenvalue for order 5.
        lambda_45: Value of the fourth deviatoric eigenvalue for order 5.
        lambda_55: Value of the fifth deviatoric eigenvalue for order 5.
        val1: Multiplicative factor for the first-order term.
        val2: Multiplicative factor for the second-order term.
        val3: Multiplicative factor for the third-order term.
        val4: Multiplicative factor for the fourth-order term.
        val5: Multiplicative factor for the fifth-order term.
        theta_1: Local angle of the first tissue layer.
        theta_2: Local angle of the second tissue layer.
        theta_3: Local angle of the third tissue layer.
        sigma_1: Local standard deviation of the first tissue layer.
        sigma_2: Local standard deviation of the second tissue layer.
        sigma_3: Local standard deviation of the third tissue layer.
        density: Local density of the tissue.
        sxx: Pointer to a double, for storing the value of the xx stress.
        syy: Pointer to a double, for storing the value of the yy stress.
        sxy: Pointer to a double, for storing the value of the xy stress.

    Returns:
        The index of the pixel that was just computed, as well as the stress
        values for xx, yy and xy.
    """

    # The shared library cannot be passed as argument and must be loaded here
    lib = ctypes.CDLL(str(lib_path))

    lib.calc_stress(*map(ctypes.c_double, (exx,
                                           eyy,
                                           exy,
                                           lambda_h,
                                           lambda_11,
                                           lambda_21,
                                           lambda_31,
                                           lambda_41,
                                           lambda_51,
                                           lambda_12,
                                           lambda_22,
                                           lambda_32,
                                           lambda_42,
                                           lambda_52,
                                           lambda_13,
                                           lambda_23,
                                           lambda_33,
                                           lambda_43,
                                           lambda_53,
                                           lambda_14,
                                           lambda_24,
                                           lambda_34,
                                           lambda_44,
                                           lambda_54,
                                           lambda_15,
                                           lambda_25,
                                           lambda_35,
                                           lambda_45,
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
                                           density)),
                    *map(ctypes.byref, (sxx, syy, sxy)))
    return i, sxx.value, syy.value, sxy.value


def _wrapper(args: tuple[Path, int, float, float, float, float, float, float,
                         float, float, float, float, float, float, float,
                         float, float, float, float, float, float, float,
                         float, float, float, float, float, float, float,
                         float, float, float, float, float, float, float,
                         float, float, float, float, float, float, float,
                         ctypes.c_double, ctypes.c_double, ctypes.c_double]
             ) -> tuple[int, float, float, float]:
    """Wrapper for passing the arguments separately to the worker, for better
    clarity.

    Args:
        args: The arguments to pass to the worker, as a single tuple.

    Returns:
        The index of the pixel that was just computed, as well as the stress
        values for xx, yy and xy.
    """

    return _worker(*args)


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
    """Computes the stress on all the specified pixels using a ProcessPool for
    efficient parallelization.

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

    # The arguments need to be formatted in a specific way to be passed to the
    # Process pool
    nb_tot = prod(exx.shape)
    mem_buf = ((ctypes.c_double(),) * nb_tot,
               (ctypes.c_double(),) * nb_tot,
               (ctypes.c_double(),) * nb_tot)
    args = zip(itertools.repeat(lib_path, nb_tot),
               range(nb_tot),
               exx.flatten(),
               eyy.flatten(),
               exy.flatten(),
               itertools.repeat(lambda_h, nb_tot),
               itertools.repeat(lambda_11, nb_tot),
               itertools.repeat(lambda_21, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(lambda_51, nb_tot),
               itertools.repeat(lambda_12, nb_tot),
               itertools.repeat(lambda_22, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(lambda_52, nb_tot),
               itertools.repeat(lambda_13, nb_tot),
               itertools.repeat(lambda_23, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(lambda_53, nb_tot),
               itertools.repeat(lambda_14, nb_tot),
               itertools.repeat(lambda_24, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(lambda_54, nb_tot),
               itertools.repeat(lambda_15, nb_tot),
               itertools.repeat(lambda_25, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(1.0, nb_tot),
               itertools.repeat(lambda_55, nb_tot),
               itertools.repeat(val1, nb_tot),
               itertools.repeat(val2, nb_tot),
               itertools.repeat(val3, nb_tot),
               itertools.repeat(val4, nb_tot),
               itertools.repeat(val5, nb_tot),
               theta_1.flatten(),
               np.nan_to_num(theta_2).flatten(),
               np.nan_to_num(theta_3).flatten(),
               sigma_1.flatten(),
               np.nan_to_num(sigma_2).flatten(),
               np.nan_to_num(sigma_3).flatten(),
               density.flatten(),
               *mem_buf)

    stress = np.empty((*exx.shape, 3), dtype=np.float64)

    # Iterating over all the pixels, and storing the stress values in the
    # dedicated arrays
    with tqdm(total=nb_tot,
              desc='Compute the stress for one image',
              file=sys.stdout,
              colour='green',
              position=2,
              leave=False) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, sxx, syy, sxy in executor.map(_wrapper, args,
                                                 chunksize=2000):
                stress[np.unravel_index(i, exx.shape)] = (sxx, syy, sxy)
                pbar.update()

    return stress[..., 0], stress[..., 1], stress[..., 2]
