# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import re

from .. import optimize_diagonals, kelvin_lib_path

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
    x0 = np.concatenate((np.array((0.00043865203019722535,
                                   27.520274700417893),
                                  dtype=np.float64),
                         np.array((0.2495873752929645,
                                   1.3366019501003517,
                                   1.9457401277770343),
                                  dtype=np.float64),
                         np.tile(np.array((0.8783740443867915,
                                           0.14472992624487574,
                                           1.010716786369869),
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
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/13_4018.npy"),
                        Path("/home/weis/Desktop/HDR/7LX1_2/hdr/14_4196.npy"))
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
    efforts_y = tuple(0.0 for _ in efforts_x)

    # The multiplicative factors to apply to each order of the model
    order_coeffs = np.array((1.0, 0.0, 1.0, 0.0, 0.0))

    # Parameters measured during the test
    scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    nb_interp_diag = ref_img.shape[0]
    interp_strain = True
    diagonal_downscaling = 20
    verbose = False
    dest_file = Path("/home/weis/Desktop/HDR/results_multi.txt")

    # Optimize an all the images at once
    optimize_diagonals(lib_path,
                       ref_img,
                       density_base,
                       gauss_fit,
                       peaks,
                       x0,
                       def_images,
                       efforts_x,
                       efforts_y,
                       order_coeffs,
                       scale,
                       thickness,
                       nb_interp_diag,
                       interp_strain,
                       diagonal_downscaling,
                       verbose,
                       dest_file)
