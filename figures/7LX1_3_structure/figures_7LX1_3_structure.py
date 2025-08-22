# coding: utf-8

import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Desktop/HDR/7LX1_3/hdr/7_04362.npy')
    sigma = 2

    plt.rcParams['font.family'] = 'serif'

    img = 1.0 - (img - img.min()) / (img.max() - img.min())

    plt.figure()
    ax = plt.gca()
    im = plt.imshow(img, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/7LX1_3_structure/image.svg', dpi=300)

    arr, arc, acc = structure_tensor(img, sigma=sigma, mode='constant', cval=0,
                                     order='rc')

    l1, l2 = structure_tensor_eigenvalues([arr, arc, acc])
    denominator = l1 + l2
    fractional_anisotropy = np.zeros_like(l1)
    valid_mask = np.abs(denominator) > 1.0e-10
    fractional_anisotropy[valid_mask] = ((l1[valid_mask] - l2[valid_mask]) /
                                         denominator[valid_mask])
    angle = np.rad2deg(0.5 * np.arctan2(2 * arc, acc - arr)) % 180

    plt.figure()
    ax = plt.gca()
    im = plt.imshow(angle, cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/7LX1_3_structure/dominant_angle.svg', dpi=300)

    plt.figure()
    ax = plt.gca()
    data = fractional_anisotropy
    im = plt.imshow(data, cmap='plasma', clim=(np.percentile(data, 1),
                                               np.percentile(data, 99)))
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Fractional anisotropy')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/7LX1_3_structure/fractional_anisotropy.svg', dpi=300)

    plt.figure()
    plt.hist(angle.flatten(),
             weights=np.full_like(angle.flatten(),
                                  1 / (angle.shape[0] * angle.shape[1])),
             bins=45)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Fraction of values')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/7LX1_3_structure/angle_distribution.svg', dpi=300)

    plt.show()
