# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import sys

from kelvin_model import kelvin_lib_path, optimize_diagonals

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
    include_divergence = False
    dest_file = Path("/home/weis/Desktop/HDR/7LX1_2/results/results.csv")
    
    for img, f_x, index in tqdm(zip(def_images, efforts_x, indexes),
                                total=len(def_images),
                                desc='Perform optimization for all the images',
                                file=sys.stdout,
                                colour='green',
                                position=0,
                                leave=False):
        optimize_diagonals(lib_path,
                           ref_img.astype(np.float64),
                           density_base.astype(np.float64),
                           gauss_fit.astype(np.float64),
                           peaks.astype(np.float64),
                           x0.astype(np.float64),
                           (img.astype(np.float64),),
                           (f_x,),
                           order_coeffs,
                           scale,
                           thickness,
                           nb_interp_diag,
                           diagonal_downscaling,
                           include_divergence,
                           verbose,
                           dest_file,
                           index)
