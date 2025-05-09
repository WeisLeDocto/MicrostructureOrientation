# coding: utf-8

from pathlib import Path
from itertools import batched, islice

from .exposure_fusion import exposure_fusion
from .gabor_filter import apply_gabor_filter
from .peak_detection import detect_peaks
from .gaussian_fit import gaussian_fit
from .make_plots import make_plots

if __name__ == '__main__':

    # 7LX1
    # roi_y = slice(1486, 3005, 1)
    # roi_x = slice(1360, 2416, 1)

    # 7LX1_2
    # roi_y = slice(1480, 2897, 1)
    # roi_x = slice(698, 1898, 1)

    # 7LX1_3
    # roi_y = slice(1529, 3042, 1)
    # roi_x = slice(1229, 2190, 1)

    # 7LX1_4
    roi_y = slice(1538, 3089, 1)
    roi_x = slice(1041, 1834, 1)

    # Parameters driving the image processing
    base_path = Path('/home/weis/Desktop/HDR/7LX1_4')
    n_images = 8
    nb_ang = 45
    nb_pix = 15

    # Location of the various folders where to store the results
    images_path = base_path / 'images'
    hdr_path = base_path / 'hdr'
    gabor_path = base_path / 'gabor'
    peak_path = base_path / 'peaks'
    fit_path = base_path / 'fit'
    anim_path = base_path / 'anim'

    # Extract the raw images in batches, and discard the first image of each
    # batch
    images = tuple(batched(sorted(images_path.glob('*.npy')), n_images))
    images = tuple(zip(*islice(zip(*images), 1, None)))

    # Create exposure fusion images from batches of raw images at different
    # exposures
    exposure_fusion(images, hdr_path, (roi_x, roi_y))

    # Compute the angular response to Gabor filters on the exposure fusion
    # images, to detect fibers and their directions
    apply_gabor_filter(hdr_path, gabor_path, nb_pix)

    # Detect peaks in the angular response to Gabor filter
    detect_peaks(gabor_path, peak_path)

    # Fit periodic gaussians on the angular response to the Gabor filter
    gaussian_fit(peak_path, gabor_path, fit_path)

    # Generate animated plots to visualize the output of the computation
    make_plots(hdr_path, gabor_path, anim_path)
