# coding: utf-8

import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/cross.npy')
    sigma = 10

    img = (img - img.min()) / (img.max() - img.min())

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
    plt.imshow(img, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/image.svg', dpi=300)

    plt.figure()
    plt.imshow(img[450:550, 450:550], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/image_close.svg', dpi=300)

    plt.figure()
    plt.imshow(angle, cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angle[450:550, 450:550], cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/dominant_angle_close.svg', dpi=300)

    plt.figure()
    plt.imshow(fractional_anisotropy, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Fractional anisotropy')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/fractional_anisotropy.svg',
                dpi=300)

    plt.figure()
    plt.imshow(fractional_anisotropy[450:550, 450:550], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Fractional anisotropy')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross_structure/fractional_anisotropy_close.svg',
                dpi=300)

    plt.close('all')
