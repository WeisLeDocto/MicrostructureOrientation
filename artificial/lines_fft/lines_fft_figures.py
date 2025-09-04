# coding: utf-8

import cupy as cp
import numpy as np
import cv2
from numpy.fft import fftshift, fft2
from scipy.signal.windows import hann
import os
from matplotlib import pyplot as plt
from microstructure_orientation.peak_detection import _find_peaks_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/lines.npy')
    filter_wavelength = 100
    sigma_x = 12
    sigma_y = 30

    img = (img - img.min()) / (img.max() - img.min())

    win = 201
    step = 10
    ntheta = 45
    H, W = img.shape
    thetas = np.linspace(0, np.pi, ntheta, endpoint=False)
    win2d = np.outer(hann(win), hann(win))
    # coords in frequency plane
    ky = np.fft.fftfreq(win)
    kx = np.fft.fftfreq(win)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    ang = (np.arctan2(KY, KX) + np.pi) % np.pi  # [0, π)
    bins = (ang * ntheta / np.pi).astype(int).clip(0, ntheta - 1)

    spectra = []
    ys = range(win // 2, H - win // 2, step)
    xs = range(win // 2, W - win // 2, step)
    for y in ys:
        row = []
        for x in xs:
            patch = img[y - win // 2:y + win // 2 + 1,
                        x - win // 2:x + win // 2 + 1] * win2d
            F = fftshift(fft2(patch))
            P = np.abs(F) ** 2
            hist = np.bincount(bins.ravel(), weights=P.ravel(),
                               minlength=ntheta)
            row.append(hist / (hist.sum() + 1e-12))
        spectra.append(np.stack(row, 0))
    spectra = np.stack(spectra, 0)

    res = cv2.resize(spectra, (1000, 1000))

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
                'artificial/lines_fft/image.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 1], cmap='twilight', clim=(0, 180))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/second_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(params[..., 0], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Standard deviation')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/std.svg', dpi=300)

    plt.figure()
    plt.imshow(params[..., 1], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Amplitude')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/intensity_indicator.svg', dpi=300)

    plt.figure()
    plt.hist(angles[..., 0].flatten(),
             weights=np.full_like(angles[..., 0].flatten(),
                                  1 / (angles.shape[0] * angles.shape[1])),
             bins=45, range=(0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Fraction of values')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/angle_histogram.svg', dpi=300)

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
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[507, 499])
    plt.vlines(angles[507, 499, 0],
               res[507, 499, int(round(angles[507, 499, 0] * 44 / 180, 0))]
               - 0.07,
               res[507, 499, int(round(angles[507, 499, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.subplot(133)
    plt.title('C')
    plt.ylim((0.1, 0.9))
    plt.yticks([])
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[669, 750],
             label='Angular response')
    plt.vlines(angles[669, 750, 0],
               res[669, 750, int(round(angles[669, 750, 0] * 44 / 180, 0))]
               - 0.07,
               res[669, 750, int(round(angles[669, 750, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.vlines(angles[669, 750, 1],
               res[669, 750, int(round(angles[669, 750, 1] * 44 / 180, 0))]
               - 0.07,
               res[669, 750, int(round(angles[669, 750, 1] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5, label='Detected peaks')
    plt.legend()
    fig.supxlabel('Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/lines_fft/angular_distributions.svg', dpi=300)

    plt.close('all')
