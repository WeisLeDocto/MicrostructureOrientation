# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from phasepack.phasecongmono import phasecongmono

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/lines.npy')

    plt.rcParams['font.family'] = 'serif'

    img = (img - img.min()) / (img.max() - img.min())

    fractional_anisotropy, angle, *_ = phasecongmono(img, minWaveLength=10)
    angle = (angle + 90) % 180

    plt.figure()
    plt.imshow(img, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_congruency/image.svg', dpi=300)

    plt.figure()
    plt.imshow(angle, cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_congruency/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(fractional_anisotropy, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Fractional anisotropy')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_congruency/fractional_anisotropy.svg',
                dpi=300)

    plt.close('all')
