# coding: utf-8

import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.filters import frangi
from skimage.feature import hessian_matrix

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/lines.npy')

    plt.rcParams['font.family'] = 'serif'

    img = (img - img.min()) / (img.max() - img.min())

    sigmas = range(20)
    vesselness = frangi(img, sigmas=sigmas, black_ridges=False)
    hrr, hrc, hcc = hessian_matrix(img, sigma=max(sigmas), order='rc')
    angle = np.rad2deg((0.5 * np.arctan2(2 * hrc, hcc - hrr)
                        + np.pi / 2) % np.pi) % 180

    plt.figure()
    plt.imshow(img, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_frangi/image.svg', dpi=300)

    plt.figure()
    plt.imshow(angle, cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_frangi/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(vesselness, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Vesselness')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_frangi/vesselness.svg',
                dpi=300)

    plt.close('all')
