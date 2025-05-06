# coding: utf-8

from pathlib import Path
from itertools import batched, islice

from exposure_fusion import exposure_fusion
from gabor_filter import apply_gabor_filter
from peak_detection import detect_peaks
from gaussian_fit import gaussian_fit
from make_plots import make_plots

if __name__ == '__main__':

    base_path = Path('/home/weis/Desktop/HDR/7LX1_3')
    n_images = 8

    # 7LX1
    # roi_y = slice(1486, 3005, 1)
    # roi_x = slice(1360, 2416, 1)

    # 7LX1_2
    # roi_y = slice(1480, 2897, 1)
    # roi_x = slice(698, 1898, 1)

    # 7LX1_3
    roi_y = slice(1529, 3042, 1)
    roi_x = slice(1229, 2190, 1)

    nb_ang = 45
    nb_pix = 15

    images_path = base_path / 'images'
    hdr_path = base_path / 'hdr'
    gabor_path = base_path / 'gabor'
    peak_path = base_path / 'peaks'
    fit_path = base_path / 'fit'
    anim_path = base_path / 'anim'

    images = tuple(batched(sorted(images_path.glob('*.npy')), n_images))
    images = tuple(zip(*islice(zip(*images), 1, None)))

    exposure_fusion(images, hdr_path, (roi_x, roi_y))

    apply_gabor_filter(hdr_path, gabor_path, nb_pix)

    detect_peaks(gabor_path, peak_path)

    gaussian_fit(peak_path, gabor_path, fit_path)

    make_plots(hdr_path, gabor_path, anim_path)
