# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import sys
import concurrent.futures
import itertools
from collections.abc import Sequence
from multiprocessing import Manager

from kelvin_model import kelvin_lib_path, optimize_diagonals


def _process_pool_wrapper(args: tuple[Path, np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray,
                                      Sequence[np.ndarray], Sequence[float],
                                      np.ndarray, float,  float, int, int,
                                      bool, Path]) -> None:
    """Wrapper for passing the arguments separately to the worker, for better
    clarity.

    Args:
        args: The arguments to pass to the worker, as a single tuple.

    Returns:
        The computed error as a float.
    """

    optimize_diagonals(*args)


if __name__ == "__main__":

    # Reference image for the image correlation
    ref_img_pth = Path("/home/weis/Desktop/HDR/7LX1_2/hdr/4_2414.npy")
    ref_img = np.load(ref_img_pth)

    # Path to the shared library for computing the stress
    lib_path = Path(kelvin_lib_path)

    # Base image for computing the density
    dens_base_pth = Path("/home/weis/Desktop/HDR/7LX1_2/images/"
                         "2424_300_35.npy")
    density_base = np.load(dens_base_pth)
    roi_y = slice(1480, 2897, 1)
    roi_x = slice(698, 1898, 1)
    density_base = density_base[roi_x, roi_y]

    # Array containing standard deviation for each layer
    gauss_fit_path = Path("/home/weis/Desktop/HDR/7LX1_2/fit/4_2414.npy")
    gauss_fit = np.load(gauss_fit_path)

    # Array containing the dominant angle for each layer
    peaks_path = Path("/home/weis/Desktop/HDR/7LX1_2/peaks/4_2414.npz")
    peaks = np.radians(np.load(peaks_path)['arr_0'])

    # Initial guess for the optimization
    x0 = np.concatenate((np.array((0.16205160432737423,
                                   13.478607766848805),
                                  dtype=np.float64),
                         np.array((0.17899774828876716,
                                   0.7517522864162445,
                                   1.5655029619999858),
                                  dtype=np.float64),
                         np.tile(np.array((9.91759017825144,
                                           0.24952165920669747,
                                           1.0655685006767808),
                                          dtype=np.float64),
                                 4)), axis=0)

    # Images to use for the optimization
    def_images_paths = (Path("/home/weis/Desktop/HDR/7LX1_2/hdr/5_2592.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/6_2770.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/7_2948.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/8_3128.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/9_3306.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/10_3484.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/11_3662.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/12_3840.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/13_4018.npy"))
    def_images = tuple(np.load(img) for img in def_images_paths)

    # Effort measured during the test
    efforts_file = Path("/home/weis/Desktop/HDR/7LX1_2/effort_2.csv")
    effort_data = pd.read_csv(efforts_file)

    # Extract the efforts corresponding to the images to use
    indexes = tuple(int(re.match(r"(\d+).+\.npy", file.name).groups()[0])
                    for file in def_images_paths)
    efforts_x = tuple(
        effort_data['F_corr(N)'][
            (effort_data["t(s)"] > idx * 7)
            & (effort_data["t(s)"] < (idx * 7) + 5)].mean()
        for idx in indexes)

    # The multiplicative factors to apply to each order of the model
    order_coeffs = np.array((1.0, 1.0, 0.0, 0.0, 0.0))

    # Parameters measured during the test
    scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    nb_interp_diag = 200  # ref_img.shape[0]
    diagonal_downscaling = 20
    verbose = False
    dest_file = Path("/home/weis/Desktop/HDR/7LX1_2/results.csv")
    manager = Manager()
    lock = manager.RLock()
    
    nb_tot = len(def_images)
    pool_args = zip(itertools.repeat(lib_path, nb_tot),
                    itertools.repeat(ref_img.astype(np.float64), nb_tot),
                    itertools.repeat(density_base.astype(np.float64), nb_tot),
                    itertools.repeat(gauss_fit.astype(np.float64), nb_tot),
                    itertools.repeat(peaks.astype(np.float64), nb_tot),
                    itertools.repeat(x0.astype(np.float64), nb_tot),
                    tuple((img.astype(np.float64),) for img in def_images),
                    tuple((f_x,) for f_x in efforts_x),
                    itertools.repeat(order_coeffs, nb_tot),
                    itertools.repeat(scale, nb_tot),
                    itertools.repeat(thickness, nb_tot),
                    itertools.repeat(nb_interp_diag, nb_tot),
                    itertools.repeat(diagonal_downscaling, nb_tot),
                    itertools.repeat(verbose, nb_tot),
                    itertools.repeat(dest_file, nb_tot),
                    indexes,
                    itertools.repeat(lock, nb_tot))
    
    with tqdm(total=len(def_images),
              desc='Perform optimization for all the images',
              file=sys.stdout,
              colour='green',
              position=0,
              leave=False) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in executor.map(_process_pool_wrapper,
                                  pool_args, chunksize=1):
                pbar.update()
