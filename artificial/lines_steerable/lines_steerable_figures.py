# coding: utf-8

import cupy as cp
import numpy as np
import pyrtools as pt
from scipy.signal import resample
import os
from matplotlib import pyplot as plt
from microstructure_orientation.peak_detection import _find_peaks_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/lines.npy')

    img = (img - img.min()) / (img.max() - img.min())

    order = 15
    pyr = pt.pyramids.SteerablePyramidFreq(img, order=order,
                                           is_complex=True)
    n_orient = order + 1
    E = np.zeros((*img.shape, n_orient), dtype=np.float32)
    for b in range(n_orient):
        recon_b = pyr.recon_pyr(levels='all',
                                bands=[b])  # full-res from just band b
        E[..., b] = np.abs(recon_b) ** 2
    E /= (E.sum(axis=-1, keepdims=True) + 1e-12)
    res = resample(E, 45, axis=-1)  # periodic FFT resample

    mem_pool = cp.get_default_memory_pool()
    angles, params = _find_peaks_gpu(res, np.linspace(0, 180, NB_ANGLES))
    mem_pool.free_all_blocks()

    plt.rcParams['font.family'] = 'serif'

    plt.figure()
    plt.imshow(img, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/image.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 1], cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/second_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(params[..., 0], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Standard deviation')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/std.svg', dpi=300)

    plt.figure()
    plt.imshow(params[..., 1], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Amplitude')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/amplitude.svg',
                dpi=300)

    plt.figure()
    plt.hist(angles[..., 0].flatten(),
             weights=np.full_like(angles[..., 0].flatten(),
                                  1 / (angles.shape[0] * angles.shape[1])),
             bins=45, range=(0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Fraction of values')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/angle_histogram.svg', dpi=300)

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title('A')
    plt.ylim((0.1, 0.9))
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[279, 291])
    plt.vlines(angles[279, 291, 0],
               res[279, 291, int(round(angles[279, 291, 0] * 44 / 180, 0))]
               - 0.07,
               res[279, 291, int(round(angles[279, 291, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.subplot(132)
    plt.title('B')
    plt.ylim((0.1, 0.9))
    plt.yticks([])
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[273, 332])
    plt.vlines(angles[273, 332, 0],
               res[273, 332, int(round(angles[273, 332, 0] * 44 / 180, 0))]
               - 0.07,
               res[273, 332, int(round(angles[273, 332, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.subplot(133)
    plt.title('C')
    plt.ylim((0.1, 0.9))
    plt.yticks([])
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[691, 750],
             label='Angular response')
    plt.vlines(angles[691, 750, 0],
               res[691, 750, int(round(angles[691, 750, 0] * 44 / 180, 0))]
               - 0.07,
               res[691, 750, int(round(angles[691, 750, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.vlines(angles[691, 750, 1],
               res[691, 750, int(round(angles[691, 750, 1] * 44 / 180, 0))]
               - 0.07,
               res[691, 750, int(round(angles[691, 750, 1] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5, label='Detected peaks')
    plt.legend()
    fig.supxlabel('Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_steerable/angular_distributions.svg',
                dpi=300)

    plt.close('all')
