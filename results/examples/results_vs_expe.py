# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import re

from kelvin_model import kelvin_lib_path, compare_results_expe

if __name__ == '__main__':

    img = '7LX1_3'

    ref_img_pth = {'7LX1': Path(f"./../data/{img}/hdr/14_5391.npy"),
                   '7LX1_2': Path(f"./../data/{img}/hdr/4_2414.npy"),
                   '7LX1_3': Path(f"./../data/{img}/hdr/7_04362.npy")}
    ref_img = np.load(ref_img_pth[img])

    lib_path = Path(kelvin_lib_path)

    density_base = np.load(Path(f"./../data/{img}/density.npy"))

    if img == '7LX1':
        roi_y = slice(1486, 3005, 1)
        roi_x = slice(1360, 2416, 1)
    elif img == '7LX1_2':
        roi_y = slice(1480, 2897, 1)
        roi_x = slice(698, 1898, 1)
    elif img == '7LX1_3':
        roi_y = slice(1529, 3042, 1)
        roi_x = slice(1229, 2190, 1)
    else:
        raise ValueError
    density_base = density_base[roi_x, roi_y]

    gauss_fit = np.load(Path(f"./../data/{img}/fit.npy"))

    peaks = np.radians(np.load(Path(f"./../data/{img}/angle.npy")))

    # Images to use for the optimization
    def_images_paths = {
        '7LX1': (Path(f"./../data/{img}/hdr/15_5570.npy"),
                 Path(f"./../data/{img}/hdr/16_5748.npy"),
                 Path(f"./../data/{img}/hdr/17_5927.npy"),
                 Path(f"./../data/{img}/hdr/18_6106.npy"),
                 Path(f"./../data/{img}/hdr/19_6286.npy"),
                 Path(f"./../data/{img}/hdr/20_6452.npy"),
                 Path(f"./../data/{img}/hdr/21_6630.npy")),
        '7LX1_2': (Path(f"./../data/{img}/hdr/5_2592.npy"),
                   Path(f"./../data/{img}/hdr/6_2770.npy"),
                   Path(f"./../data/{img}/hdr/7_2948.npy"),
                   Path(f"./../data/{img}/hdr/8_3128.npy"),
                   Path(f"./../data/{img}/hdr/9_3306.npy"),
                   Path(f"./../data/{img}/hdr/10_3484.npy"),
                   Path(f"./../data/{img}/hdr/11_3662.npy"),
                   Path(f"./../data/{img}/hdr/12_3840.npy"),
                   Path(f"./../data/{img}/hdr/13_4018.npy")),
        '7LX1_3': (Path(f"./../data/{img}/hdr/8_04540.npy"),
                   Path(f"./../data/{img}/hdr/9_04718.npy"),
                   Path(f"./../data/{img}/hdr/10_04897.npy"),
                   Path(f"./../data/{img}/hdr/11_05075.npy"),
                   Path(f"./../data/{img}/hdr/12_05254.npy"),
                   Path(f"./../data/{img}/hdr/13_05432.npy"),
                   Path(f"./../data/{img}/hdr/14_05598.npy"),
                   Path(f"./../data/{img}/hdr/15_05777.npy"),
                   Path(f"./../data/{img}/hdr/16_05955.npy"))}

    def_images = tuple(np.load(image) for image in def_images_paths[img])

    effort_data = pd.read_csv(Path(f"./../data/{img}/effort_2.csv"))

    # Extract the efforts corresponding to the images to use
    indexes = tuple(int(re.match(r"(\d+).+\.npy", file.name).groups()[0])
                    for file in def_images_paths[img])
    efforts_x = tuple(
        effort_data['F_corr(N)'][
            (effort_data["t(s)"] > idx * 7)
            & (effort_data["t(s)"] < (idx * 7) + 5)].mean()
        for idx in indexes)

    times = tuple(idx * 7 for idx in indexes)

    position_data = pd.read_csv(Path(f"./../data/{img}/position_2.csv"))

    positions = tuple(
        position_data['position1_corr'][
            (position_data["t(s)"] > idx * 7)
            & (position_data["t(s)"] < (idx * 7) + 5)].mean() +
        position_data['position2_corr'][
            (position_data["t(s)"] > idx * 7)
            & (position_data["t(s)"] < (idx * 7) + 5)].mean()
        for idx in indexes)

    # Parameters measured during the test
    scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    nb_interp_diag = 200  # ref_img.shape[0]
    diagonal_downscaling = 20
    include_divergence = False
    results_file = Path(f"./../data/{img}/results.csv")
    dest_file = Path(f"./../data/{img}/comparison.csv")

    compare_results_expe(lib_path,
                         ref_img.astype(np.float64),
                         density_base.astype(np.float64),
                         gauss_fit.astype(np.float64),
                         peaks.astype(np.float64),
                         results_file,
                         def_images,
                         times,
                         efforts_x,
                         positions,
                         scale,
                         thickness,
                         nb_interp_diag,
                         diagonal_downscaling,
                         dest_file)
