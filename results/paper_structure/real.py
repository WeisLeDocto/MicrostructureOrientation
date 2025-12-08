# coding: utf-8

import cupy as cp
import numpy as np
import cucim.skimage.filters as gpu_filters
import cupyx.scipy.signal as gpu_signal
from numba import cuda
import os
import sys
import math
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
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
    sign_3 = amp * cp.exp(-cp.power(cp.arctan2(cp.sin(2 * (sign_3 - ang)),
                                               cp.cos(2 * (sign_3 - ang)))
                                    / (4 * sig), 2))
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
    min_p, max_p = np.percentile(anisotropy, 1), np.percentile(anisotropy, 99)
    anisotropy = (anisotropy - min_p) / (max_p - min_p)
    anisotropy = np.clip(anisotropy, 0, 1)
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

    fig = plt.figure(figsize=(11, 7))

    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    image = np.load('../../results/data/7LX1_3/hdr/7_04362.npy')
    image = 1.0 - (image - image.min()) / (image.max() - image.min())

    (raw, angles, _, _, score, aniso, _) = process_img(image, 60.0, 4, 20, 25)

    ax1.imshow(image, cmap='Greys', clim=(0, 1))
    ax1.set_title('(a)', loc='left')
    ax1.set_xticklabels([])

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Greys'),
                 ax=[ax1], label='Normalized pixel value')

    ax2.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))
    ax2.set_title('(b)', loc='left')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=[ax2], label='Angle (degrees)')

    data = np.clip((aniso[..., 0] - np.percentile(aniso[..., 0], 1)) /
                   (np.percentile(aniso[..., 0], 99) -
                    np.percentile(aniso[..., 0], 1)), 0, 1)
    ax3.imshow(data, cmap='magma', clim=(0, 1))
    ax3.set_title('(c)', loc='left')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=[ax3],
                 label='Fiber-likeness score')

    ax4.imshow(score[..., 0], cmap='magma', clim=(0, 1))
    ax4.set_yticklabels([])
    ax4.set_title('(d)', loc='left')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'),
                 ax=[ax4], label='Layer score')

    ax5.hist(angles[~np.isnan(angles)].flatten(),
             weights=aniso[~np.isnan(angles)].flatten() /
             np.sum(aniso[~np.isnan(angles)]) * 100,
             bins=45)
    ax5.set_xlabel('Angle (degrees)')
    ax5.set_ylabel('% values')
    ax5.set_title('(e)', loc='left')

    data = raw[215, 591] - np.min(raw[215, 591])
    ax6.plot(np.linspace(0, 180, NB_ANGLES), data / np.sum(data))
    ax6.set_xlabel('Angle (degrees)')
    ax6.set_ylabel('Filter response')
    ax6.yaxis.set_label_position("right")
    ax6.yaxis.tick_right()
    ax6.set_title('(f)', loc='left')

    plt.savefig('./real_7LX1_3.svg', dpi=300, transparent=True)

    plt.show()

    fig = plt.figure(figsize=(12, 12))

    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    image = np.load('../data/composite.npy')
    image = 1.0 - (image - image.min()) / (image.max() - image.min())

    (raw, angles, _, _, score, aniso, _) = process_img(image, 60.0, 2, 10, 25)

    ax1.imshow(image, cmap='Greys', clim=(0, 1))
    ax1.set_title('(a)', loc='left')
    ax1.set_xticklabels([])

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Greys'),
                 ax=[ax1], label='Normalized pixel value')

    ax2.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))
    ax2.set_title('(b)', loc='left')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=[ax2], label='Angle (degrees)')

    data = np.clip((aniso[..., 0] - np.percentile(aniso[..., 0], 1)) /
                   (np.percentile(aniso[..., 0], 99) -
                    np.percentile(aniso[..., 0], 1)), 0, 1)
    ax3.imshow(data, cmap='magma', clim=(0, 1))
    ax3.set_title('(c)', loc='left')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=[ax3],
                 label='Fiber-likeness score')

    ax4.imshow(score[..., 0], cmap='magma', clim=(0, 1))
    ax4.set_yticklabels([])
    ax4.set_title('(d)', loc='left')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'),
                 ax=[ax4], label='Layer score')

    ax5.hist(angles[~np.isnan(angles)].flatten(),
             weights=aniso[~np.isnan(angles)].flatten() /
             np.sum(aniso[~np.isnan(angles)]) * 100,
             bins=45)
    ax5.set_xlabel('Angle (degrees)')
    ax5.set_ylabel('% values')
    ax5.set_title('(e)', loc='left')

    data = raw[566, 425] - np.min(raw[566, 425])
    ax6.plot(np.linspace(0, 180, NB_ANGLES), data / np.sum(data))
    ax6.set_xlabel('Angle (degrees)')
    ax6.set_ylabel('Filter response')
    ax6.set_title('(f)', loc='left')

    plt.savefig('./real_composite.svg', dpi=300)

    plt.show()

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 10),
                            layout='constrained')

    image = np.load('../data/composite.npy')
    image = 1.0 - (image - image.min()) / (image.max() - image.min())

    (raw, angles, _, _, score, aniso, _) = process_img(image, 60.0, 4, 20, 25)

    plt.subplot(2, 2, 1)
    plt.imshow(image[400:800, 400:800], cmap='Greys', clim=(0, 1))
    plt.title('(a)', loc='left')

    plt.subplot(2, 2, 2)

    data = np.full_like(angles[..., 0], np.nan)
    data[(angles[..., 0] >= 160) | (angles[..., 0] <= 20)] = 0
    data[(angles[..., 1] >= 160) | (angles[..., 1] <= 20)] = 1

    cmap = plt.get_cmap('rainbow', 2)
    plt.imshow(data[400:800, 400:800], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(b)', loc='left')
    plt.xlabel(r'$\theta < 20°, \quad \theta > 160°$')

    plt.subplot(2, 2, 3)

    data = np.full_like(angles[..., 0], np.nan)
    data[(angles[..., 0] >= 70) & (angles[..., 0] <= 110)] = 0
    data[(angles[..., 1] >= 70) & (angles[..., 1] <= 110)] = 1

    plt.imshow(data[400:800, 400:800], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(c)', loc='left')
    plt.xlabel(r'$70° < \theta < 110°$')

    plt.subplot(2, 2, 4)

    data = np.full_like(angles[..., 0], np.nan)
    data[((angles[..., 0] >= 20) & (angles[..., 0] <= 70)) |
         ((angles[..., 0] >= 110) & (angles[..., 0] <= 160))] = 0
    data[((angles[..., 1] >= 20) & (angles[..., 1] <= 70)) |
         ((angles[..., 1] >= 110) & (angles[..., 1] <= 160))] = 1

    plt.imshow(data[400:800, 400:800], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(d)', loc='left')
    plt.xlabel(r'$20° < \theta < 70°, \quad 110° < \theta < 160°$')

    plt.savefig('./layer_composite.svg', dpi=300)

    plt.show()

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 10),
                            layout='constrained')

    image = np.load('../../results/data/7LX1_3/hdr/7_04362.npy')
    image = 1.0 - (image - image.min()) / (image.max() - image.min())

    (raw, angles, _, _, score, aniso, _) = process_img(image, 60.0, 4, 20, 25)

    plt.subplot(2, 2, 1)
    plt.imshow(image[:600, :600], cmap='Greys', clim=(0, 1))
    plt.title('(a)', loc='left')

    plt.subplot(2, 2, 2)

    data = np.full_like(angles[..., 0], np.nan)
    data[(angles[..., 0] >= 40) & (angles[..., 0] <= 80)] = 0
    data[(angles[..., 1] >= 40) & (angles[..., 1] <= 80)] = 1

    cmap = plt.get_cmap('rainbow', 2)
    plt.imshow(data[:600, :600], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(b)', loc='left')
    plt.xlabel(r'$40° < \theta < 80°$')

    plt.subplot(2, 2, 3)

    data = np.full_like(angles[..., 0], np.nan)
    data[(angles[..., 0] >= 105) & (angles[..., 0] <= 145)] = 0
    data[(angles[..., 1] >= 105) & (angles[..., 1] <= 145)] = 1

    plt.imshow(data[:600, :600], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(c)', loc='left')
    plt.xlabel(r'$105° < \theta < 145°$')

    plt.subplot(2, 2, 4)

    data = np.full_like(angles[..., 0], np.nan)
    data[(angles[..., 0] <= 40) | (angles[..., 0] >= 145) |
         ((angles[..., 0] >= 80) & (angles[..., 0] <= 105))] = 0
    data[(angles[..., 1] <= 40) | (angles[..., 1] >= 145) |
         ((angles[..., 1] >= 80) & (angles[..., 1] <= 105))] = 1

    plt.imshow(data[:600, :600], cmap=cmap, clim=(0, 1))
    plt.legend([mpatches.Patch(color=cmap(i)) for i in range(2)],
               ['Layer 1', 'Layer 2'])
    plt.title('(d)', loc='left')
    plt.xlabel(r'$\theta < 40°, \quad 80° < \theta < 105°, '
               r'\quad \theta > 145°$')

    plt.savefig('./layer_7LX1_3.svg', dpi=300)

    plt.show()
