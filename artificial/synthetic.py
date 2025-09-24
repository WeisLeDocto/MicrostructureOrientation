# coding: utf-8

import cupy as cp
import numpy as np
import cucim.skimage.filters as gpu_filters
import cupyx.scipy.signal as gpu_signal
from numba import cuda
import os
import sys
import math
import string
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from microstructure_orientation.peak_detection import _find_peaks_gpu
from microstructure_orientation.gaussian_fit import _fit_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def process_img(img: np.ndarray,
                filter_wavelength: float,
                sigma_x: float,
                sigma_y: float,
                gate_thresh: float) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray,
                                             np.ndarray]:
    """"""

    img = (img - img.min()) / (img.max() - img.min())

    kernels = {i: gpu_filters.gabor_kernel(frequency=1 / filter_wavelength,
                                           theta=np.pi / 2 - ang,
                                           n_stds=2,
                                           offset=0,
                                           dtype=cp.complex64,
                                           sigma_x=sigma_x,
                                           sigma_y=sigma_y)
               for i, ang
               in enumerate(np.linspace(0, np.pi, NB_ANGLES).tolist())}

    img_gpu = cp.asarray(img, dtype='float32')
    res_gpu = cp.zeros(shape=(*img.shape, NB_ANGLES), dtype='float32')

    for i, kernel in tqdm(kernels.items(),
                          total=NB_ANGLES,
                          desc='Apply Gabor kernels',
                          file=sys.stdout,
                          colour='green',
                          mininterval=0.001,
                          maxinterval=0.01,
                          position=0,
                          leave=True):
        conv = gpu_signal.convolve2d(img_gpu,
                                     kernel,
                                     mode='same',
                                     boundary='fill',
                                     fillvalue=0).astype(cp.complex64)
        conv /= cp.linalg.norm(cp.sum(kernel))
        res_gpu[:, :, i] = cp.sqrt(conv.real ** 2 + conv.imag ** 2)

    res = cp.asnumpy(res_gpu)

    angles, params = _find_peaks_gpu(res, np.linspace(0, 180, NB_ANGLES))

    tpb = (16, 16)
    bpg = (int(math.ceil(res.shape[0] / tpb[0])),
           int(math.ceil(res.shape[1] / tpb[1])))

    # Count the number of peaks and load in GPU memory
    n_peaks = np.count_nonzero(np.invert(np.isnan(angles)), axis=-1)
    n_gpu = cuda.to_device(n_peaks.astype(np.float32))

    # Load all the data in GPU memory
    x_gpu = cuda.to_device(
        np.radians(np.linspace(0, 180, NB_ANGLES)).astype(np.float32))
    y_gpu = cuda.to_device(res.astype(np.float32))
    p_gpu = cuda.to_device(params.astype(np.float32))
    m_gpu = cuda.to_device(np.radians(angles).astype(np.float32))

    # Perform the gaussian fit
    _fit_gpu[bpg, tpb](n_gpu, x_gpu, y_gpu, p_gpu, m_gpu, 1e-6, 5000)

    # Copy the result in CPU memory and write it on the disk
    param = p_gpu.copy_to_host()

    ang = cp.nan_to_num(
        cp.deg2rad(cp.asarray(angles, dtype=cp.float32))[..., cp.newaxis])
    amp = cp.nan_to_num(cp.asarray(np.stack((param[..., 1],
                                             param[..., 3],
                                             param[..., 5]),
                                            axis=2)[..., np.newaxis],
                                   dtype=cp.float32))
    sig = cp.nan_to_num(cp.asarray(np.stack((param[..., 0],
                                             param[..., 2],
                                             param[..., 4]),
                                            axis=2)[..., np.newaxis],
                                   dtype=cp.float32))
    amp[sig <= 0] = 0
    sig[sig <= 0] = 1.0e-5

    sign_3 = cp.tile(cp.linspace(0, cp.pi, NB_ANGLES),
                     (img.shape[0], img.shape[1], 3, 1))
    sign_3 = amp * cp.exp(-cp.power((cp.mod(sign_3 + cp.pi / 2 - ang, cp.pi)
                                     - cp.pi / 2) / sig, 2))
    signal = cp.sum(sign_3, axis=2)
    del ang

    amp = cp.squeeze(amp)
    sig = cp.squeeze(sig)

    signal_sum = cp.sum(signal, axis=2)
    amp_norm = amp / signal_sum[..., cp.newaxis]

    del signal
    sign_3_norm = cp.nan_to_num(sign_3 /
                                signal_sum[..., cp.newaxis, cp.newaxis])
    del sign_3, signal_sum
    layer_score = cp.sum(sign_3_norm, axis=3)

    low_thresh = cp.percentile(amp, gate_thresh,
                               axis=(0, 1))[cp.newaxis, cp.newaxis, :]
    low_thresh = cp.nan_to_num(low_thresh)
    thresh = 1 / (1 + cp.exp(-10 * (amp - low_thresh)))
    amp_norm *= thresh

    del low_thresh, thresh, amp

    anisotropy = np.nan_to_num(amp_norm / sig)
    anisotropy_unique = 1 - np.prod(1 - anisotropy, axis=2)

    sig = cp.asnumpy(cp.squeeze(sig))
    amp_norm = cp.asnumpy(amp_norm)
    layer_score = cp.asnumpy(layer_score)
    anisotropy = cp.asnumpy(anisotropy)
    anisotropy_unique = cp.asnumpy(anisotropy_unique)

    return (res, angles, sig, amp_norm, layer_score, anisotropy,
            anisotropy_unique)


if __name__ == "__main__":

    plt.rcParams['font.family'] = 'serif'

    images = ('./lines.npy', './lines_intensity.npy', './cross.npy',
              './circles.npy', './waves.npy')
    images = tuple(map(np.load, images))
    sigmas_x = (12, 12, 7, 12, 12)
    sigmas_y = (30, 30, 12, 30, 30)
    thresholds = (5, 5, 25, 5, 25)

    fig, axs = plt.subplots(1, 5, sharex='col', sharey='row', figsize=(14, 2),
                            layout='constrained')

    for k, img in enumerate(images):

        plt.subplot(1, 5, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})", loc='left')
        plt.imshow((img - img.min()) / (img.max() - img.min()), cmap='Greys_r')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Greys_r'),
                 ax=axs, label='Normalized pixel value')

    plt.savefig('./comp_fig.svg', dpi=300)

    plt.show()

    exit()

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 6),
                            layout='constrained')

    for k in range(2):

        (_, angles, _, _, _,
         aniso, _) = process_img(images[4], 60.0,
                                 sigmas_x[4] if k == 0 else 8,
                                 sigmas_y[4] if k == 0 else 18,
                                 thresholds[4])

        plt.subplot(2, 2, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow(angles[400:600, 400:600, 0], cmap='twilight', clim=(0, 180))

        plt.subplot(2, 2, k + 3)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        data = np.clip((aniso[400:600, 400:600, 0] -
                        np.percentile(aniso[400:600, 400:600, 0], 1)) /
                       (np.percentile(aniso[400:600, 400:600, 0], 99) -
                        np.percentile(aniso[400:600, 400:600, 0], 1)), 0, 1)
        plt.imshow(data, cmap='magma')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=axs[0], label='Angle (degrees)')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=axs[1],
                 label='Fiber-likeness score')

    plt.savefig('./comp_wave.svg', dpi=300)

    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    (_, angles, _, _, _, aniso, _) = process_img(images[3],
                                                 60.0,
                                                 sigmas_x[3],
                                                 sigmas_y[3],
                                                 thresholds[3])

    plt.subplot(1, 2, 1)
    plt.hist(angles[~np.isnan(angles)].flatten(),
             weights=aniso[~np.isnan(angles)].flatten() /
             np.sum(aniso[~np.isnan(angles)]) * 100,
             bins=45)
    plt.ylim((0, 5))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('% values')
    plt.title('(a)', loc='left')

    img = (images[3] - images[3].min()) / (images[3].max() - images[3].min())

    rows, cols = np.indices(img.shape)
    phi = np.arctan2(rows - 499, cols - 499)
    phi = np.rad2deg(np.mod(phi + np.pi / 2, np.pi))
    phi_plus = np.tile(phi, (45, 1, 1))
    diff = phi_plus - np.linspace(0, 180, 45)[:, np.newaxis, np.newaxis]
    idx_min = np.argmin(np.abs(diff), axis=0)
    idx_x = np.tile(np.arange(phi.shape[0])[:, np.newaxis], (1, phi.shape[1]))
    idx_y = np.tile(np.arange(phi.shape[1]), (phi.shape[0], 1))
    phi = phi.flatten() - diff[idx_min.flatten(), idx_x.flatten(),
                               idx_y.flatten()]

    plt.subplot(1, 2, 2)
    plt.hist(phi, bins=45,
             weights=img.flatten() / np.sum(img) * 100)
    plt.ylim((0, 5))
    plt.xlabel('Angle (degrees)')
    plt.title('(b)', loc='left')
    ax2.set_yticklabels([])

    plt.savefig('./comp_circle.svg', dpi=300)

    plt.show()

    fig = plt.figure(figsize=(10, 12))  # , layout='constrained')

    (raw, angles, _, _, score,
     aniso, _) = process_img(images[2],
                             60.0,
                             sigmas_x[2],
                             sigmas_y[2],
                             thresholds[2])

    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    ax1.imshow(angles[400:600, 400:600, 0], cmap='twilight', clim=(0, 180))
    ax1.set_title('(a)', loc='left')
    ax1.set_xticklabels([])
    ax2.imshow(angles[400:600, 400:600, 1], cmap='twilight', clim=(0, 180))
    ax2.set_title('(b)', loc='left')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ax3.imshow(score[400:600, 400:600, 0], cmap='magma', clim=(0, 1))
    ax3.set_title('(c)', loc='left')
    ax4.hist(angles[~np.isnan(angles)].flatten(),
             weights=aniso[~np.isnan(angles)].flatten() /
             np.sum(aniso[~np.isnan(angles)]) * 100,
             bins=45)
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('% values')
    ax4.set_title('(d)', loc='left')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()

    data = raw[691, 750] - np.min(raw[691, 750])
    ax5.plot(np.linspace(0, 180, NB_ANGLES), data / np.sum(data))
    ax5.set_xlabel('Angle (degrees)')
    ax5.set_ylabel('Filter response')
    ax5.set_title('(e)', loc='left')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=[ax1, ax2], label='Angle (degrees)')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'),
                 ax=[ax3], label='Layer score', use_gridspec=True)

    plt.savefig('./comp_cross.svg', dpi=300)

    plt.show()

    fig, axs = plt.subplots(3, 5, sharex='col', sharey='row', figsize=(18, 7),
                            layout='constrained')

    for k, (img, sig_x, sig_y, thresh) in enumerate(zip(images, sigmas_x,
                                                        sigmas_y, thresholds)):

        (_, angles, _, _, _, aniso, _) = process_img(img, 60.0, sig_x,
                                                     sig_y, thresh)

        plt.subplot(3, 5, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow((img - img.min()) / (img.max() - img.min()), cmap='Greys_r')

        plt.subplot(3, 5, k + 6)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        plt.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))

        plt.subplot(3, 5, k + 11)
        if k == 0:
            plt.ylabel('3', rotation='horizontal')
        data = np.clip((aniso[..., 0] - np.percentile(aniso[..., 0], 1)) /
                       (np.percentile(aniso[..., 0], 99) -
                        np.percentile(aniso[..., 0], 1)), 0, 1)
        plt.imshow(data, cmap='magma')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Greys_r'),
                 ax=axs[0], label='Normalized pixel value')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=axs[1], label='Angle (degrees)')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=axs[2],
                 label='Fiber-likeness score')

    plt.savefig('./comp_meth.svg', dpi=300)

    plt.show()
