# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import re
import cv2

from kelvin_model import optimize_diagonals, kelvin_lib_path

if __name__ == "__main__":

    img = '7LX1'

    ref_img_pth = {'7LX1': Path(f"./../data/{img}/hdr/14_5391.npy"),
                   '7LX1_2': Path(f"./../data/{img}/hdr/4_2414.npy"),
                   '7LX1_3': Path(f"./../data/{img}/hdr/7_04362.npy"),
                   '7LX1_2_2': Path(f"./../data/{img}/hdr/14_5391.npy"),
                   '7LX1_2_2_2': Path(f"./../data/{img}/hdr/4_2414.npy"),
                   '7LX1_3_2_2': Path(f"./../data/{img}/hdr/7_04362.npy")}
    ref_img = np.load(ref_img_pth[img])

    if img.endswith('_2_2'):
        ref_img = cv2.resize(ref_img, None, None, 0.5, 0.5,
                             interpolation=cv2.INTER_LANCZOS4)

    lib_path = Path(kelvin_lib_path)

    density_base = np.load(Path(f"./../data/{img}/density.npy"))

    if img in ('7LX1', '7LX1_2_2'):
        roi_y = slice(1486, 3005, 1)
        roi_x = slice(1360, 2416, 1)
    elif img in ('7LX1_2', '7LX1_2_2_2'):
        roi_y = slice(1480, 2897, 1)
        roi_x = slice(698, 1898, 1)
    elif img in ('7LX1_3', '7LX1_3_2_2'):
        roi_y = slice(1529, 3042, 1)
        roi_x = slice(1229, 2190, 1)
    else:
        raise ValueError
    density_base = density_base[roi_x, roi_y]

    if img.endswith('_2_2'):
        density_base = cv2.resize(density_base, None, None, 0.5, 0.5,
                                  interpolation=cv2.INTER_LANCZOS4)

    gauss_fit = np.load(Path(f"./../data/{img}/fit.npy"))

    peaks = np.radians(np.load(Path(f"./../data/{img}/angle.npy")))

    # Initial guess for the optimization
    if img == '7LX1':
        x0 = np.concatenate((np.array((0.7022429976982819,
                                       0.8372398247479019,
                                       16.629914900814708),
                                      dtype=np.float64),
                             np.array((0.9845826085697628,
                                       0.5214099554831811,
                                       1.6181803653343279),
                                      dtype=np.float64),
                             np.tile(np.array((24.37505554469635,
                                               17.962649088641044,
                                               25.26080663554753),
                                              dtype=np.float64),
                                     4)), axis=0)
    elif img == '7LX1_2':
        x0 = np.concatenate((np.array((0.211683750184906,
                                       0.9127219089203469,
                                       12.564758301378689),
                                      dtype=np.float64),
                             np.array((0.45893001514471277,
                                       0.5122795520677504,
                                       0.7914580592219166),
                                      dtype=np.float64),
                             np.tile(np.array((10.317946872584695,
                                               8.05016820244279,
                                               7.839143968581188),
                                              dtype=np.float64),
                                     4)), axis=0)
    elif img == '7LX1_3':
        x0 = np.concatenate((np.array((0.25886570631810696,
                                       2.5983863056219034,
                                       32.36193377754075),
                                      dtype=np.float64),
                             np.array((1.1857019045405095,
                                       1.0847766862635286,
                                       1.4170445192794126),
                                      dtype=np.float64),
                             np.tile(np.array((8.670872076886791,
                                               10.446482690869368,
                                               8.228342915473803),
                                              dtype=np.float64),
                                     4)), axis=0)
    elif img == '7LX1_2_2':
        x0 = np.concatenate((np.array((0.7645344664519325,
                                       0.7703386000277705,
                                       16.64606873211841),
                                      dtype=np.float64),
                             np.array((1.0490839147028517,
                                       0.4953397546436127,
                                       1.7054027546866348),
                                      dtype=np.float64),
                             np.tile(np.array((25.75147018984659,
                                               18.784466264367563,
                                               27.03612839192092),
                                              dtype=np.float64),
                                     4)), axis=0)
    elif img == '7LX1_2_2_2':
        x0 = np.concatenate((np.array((0.2979068170563402,
                                       0.974457412677047,
                                       12.53763951723783),
                                      dtype=np.float64),
                             np.array((0.4679208230907614,
                                       0.45620408068120616,
                                       0.7864920536240354),
                                      dtype=np.float64),
                             np.tile(np.array((10.937098284805998,
                                               7.783344419554566,
                                               9.07228095117343),
                                              dtype=np.float64),
                                     4)), axis=0)
    elif img == '7LX1_3_2_2':
        x0 = np.concatenate((np.array((0.2618673080379288,
                                       2.589168825052311,
                                       32.36323462665295),
                                      dtype=np.float64),
                             np.array((1.172929100896463,
                                       1.0685109059980624,
                                       1.415657743732887),
                                      dtype=np.float64),
                             np.tile(np.array((8.988085740473942,
                                               10.64940154944092,
                                               8.771182543858398),
                                              dtype=np.float64),
                                     4)), axis=0)
    else:
        raise ValueError

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
                   Path(f"./../data/{img}/hdr/16_05955.npy")),
        '7LX1_2_2': (Path(f"./../data/{img}/hdr/15_5570.npy"),
                     Path(f"./../data/{img}/hdr/16_5748.npy"),
                     Path(f"./../data/{img}/hdr/17_5927.npy"),
                     Path(f"./../data/{img}/hdr/18_6106.npy"),
                     Path(f"./../data/{img}/hdr/19_6286.npy"),
                     Path(f"./../data/{img}/hdr/20_6452.npy"),
                     Path(f"./../data/{img}/hdr/21_6630.npy")),
        '7LX1_2_2_2': (Path(f"./../data/{img}/hdr/5_2592.npy"),
                       Path(f"./../data/{img}/hdr/6_2770.npy"),
                       Path(f"./../data/{img}/hdr/7_2948.npy"),
                       Path(f"./../data/{img}/hdr/8_3128.npy"),
                       Path(f"./../data/{img}/hdr/9_3306.npy"),
                       Path(f"./../data/{img}/hdr/10_3484.npy"),
                       Path(f"./../data/{img}/hdr/11_3662.npy"),
                       Path(f"./../data/{img}/hdr/12_3840.npy"),
                       Path(f"./../data/{img}/hdr/13_4018.npy")),
        '7LX1_3_2_2': (Path(f"./../data/{img}/hdr/8_04540.npy"),
                       Path(f"./../data/{img}/hdr/9_04718.npy"),
                       Path(f"./../data/{img}/hdr/10_04897.npy"),
                       Path(f"./../data/{img}/hdr/11_05075.npy"),
                       Path(f"./../data/{img}/hdr/12_05254.npy"),
                       Path(f"./../data/{img}/hdr/13_05432.npy"),
                       Path(f"./../data/{img}/hdr/14_05598.npy"),
                       Path(f"./../data/{img}/hdr/15_05777.npy"),
                       Path(f"./../data/{img}/hdr/16_05955.npy"))}

    def_images = tuple(np.load(image) for image in def_images_paths[img])
    
    if img.endswith('_2_2'):
        def_images = tuple(cv2.resize(hdr, None, None, 0.5, 0.5,
                                      interpolation=cv2.INTER_LANCZOS4)
                           for hdr in def_images)

    effort_data = pd.read_csv(Path(f"./../data/{img}/effort_2.csv"))

    # Extract the efforts corresponding to the images to use
    indexes = tuple(int(re.match(r"(\d+).+\.npy", file.name).groups()[0])
                    for file in def_images_paths[img])
    efforts_x = tuple(
        effort_data['F_corr(N)'][
            (effort_data["t(s)"] > idx * 7)
            & (effort_data["t(s)"] < (idx * 7) + 5)].mean()
        for idx in indexes)

    # The multiplicative factors to apply to each order of the model
    order_coeffs = np.array((1.0, 1.0, 0.0, 0.0, 0.0))

    # Parameters measured during the test
    if img.endswith('_2_2'):
        scale = 0.02
    else:
        scale = 0.01
    thickness = 0.54

    # Other parameters driving the optimization process
    nb_interp_diag = 200
    diagonal_downscaling = 20
    verbose = True
    include_divergence = False
    dest_file = Path(f"./../data/{img}/results.csv")

    # Optimize an all the images at once
    optimize_diagonals(lib_path,
                       ref_img.astype(np.float64),
                       density_base.astype(np.float64),
                       gauss_fit.astype(np.float64),
                       peaks.astype(np.float64),
                       x0.astype(np.float64),
                       def_images,
                       efforts_x,
                       order_coeffs,
                       scale,
                       thickness,
                       nb_interp_diag,
                       diagonal_downscaling,
                       include_divergence,
                       verbose,
                       dest_file,
                       index=0)
