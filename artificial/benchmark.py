# coding: utf-8

import numpy as np
import cupy as cp
import pyrtools as pt
from numpy.fft import fftshift, fft2
from scipy.signal.windows import hann
from scipy.signal import resample
from numba import cuda
import cv2
import os
import string
import math
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import frangi
from skimage.feature import hessian_matrix
from phasepack.phasecongmono import phasecongmono
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from microstructure_orientation.peak_detection import _find_peaks_gpu
from microstructure_orientation.gaussian_fit import _fit_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def proc_structure(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    img = (img - img.min()) / (img.max() - img.min())
    arr, arc, acc = structure_tensor(img, sigma=10, mode='constant', cval=0,
                                     order='rc')

    l1, l2 = structure_tensor_eigenvalues([arr, arc, acc])
    denominator = l1 + l2
    fractional_anisotropy = np.zeros_like(l1)
    valid_mask = np.abs(denominator) > 1.0e-10
    fractional_anisotropy[valid_mask] = ((l1[valid_mask] - l2[valid_mask]) /
                                         denominator[valid_mask])
    angle = np.rad2deg(0.5 * np.arctan2(2 * arc, acc - arr)) % 180

    return angle, fractional_anisotropy


def proc_frangi(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    img = (img - img.min()) / (img.max() - img.min())

    sigmas = range(20)
    vesselness = frangi(img, sigmas=sigmas, black_ridges=False)
    hrr, hrc, hcc = hessian_matrix(img, sigma=max(sigmas), order='rc')
    angle = np.rad2deg((0.5 * np.arctan2(2 * hrc, hcc - hrr)
                        + np.pi / 2) % np.pi) % 180

    return angle, vesselness


def proc_congruency(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    img = (img - img.min()) / (img.max() - img.min())

    fractional_anisotropy, angle, *_ = phasecongmono(img, minWaveLength=10)
    angle = (angle + 90) % 180

    return angle, fractional_anisotropy


def proc_fft(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    return angles, params[..., 1], res


def proc_steerable(img: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray,
                                             np.ndarray]:

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

    low_thresh = cp.percentile(amp, 25,
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

    return (angles, anisotropy, res, sig, amp_norm, layer_score,
            anisotropy_unique)


if __name__ == "__main__":

    plt.rcParams['font.family'] = 'serif'

    images = ('./lines.npy', './lines_intensity.npy', './cross.npy',
              './waves.npy', './circles.npy')
    images = tuple(map(np.load, images))
    methods = (proc_structure, proc_frangi, proc_congruency, proc_fft,
               proc_steerable)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    for k, method in enumerate(methods[3:]):

        angles, _, res, *_ = method(images[2])

        plt.subplot(2, 2, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow(angles[400:600, 400:600, 1], cmap='twilight', clim=(0, 180))

        plt.subplot(2, 2, k + 3)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        data = res[691, 750] - res[691, 750].min()
        plt.plot(np.linspace(0, 180, NB_ANGLES), data / np.sum(data))
        plt.xlabel('Angle (degrees)')

    axs[0][1].set_yticklabels([])

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=axs[0], label='Angle (degrees)')

    plt.savefig('./benchamrk_cross_add.svg', dpi=300)

    plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))

    for k, img in enumerate(images[3:]):

        (angles, aniso, *_) = proc_steerable(img)

        plt.subplot(2, 3, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow(angles[..., 0], cmap='twilight', clim=(0, 180))

        plt.subplot(2, 3, k + 4)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        data = np.clip((aniso[..., 0] - np.percentile(aniso[..., 0], 1)) /
                       (np.percentile(aniso[..., 0], 99) -
                        np.percentile(aniso[..., 0], 1)), 0, 1)
        plt.imshow(data, cmap='magma')

    ax[0][0].set_xticklabels([])
    ax[0][1].set_xticklabels([])
    ax[0][2].set_xticklabels([])
    ax[0][1].set_yticklabels([])
    ax[1][1].set_yticklabels([])

    plt.subplot(2, 3, 3)
    plt.title(f"({string.ascii_lowercase[2]})")
    plt.hist(angles[~np.isnan(angles)].flatten(),
             weights=aniso[~np.isnan(angles)].flatten() /
             np.sum(aniso[~np.isnan(angles)]) * 100,
             bins=45)
    plt.ylim((0, 5))
    plt.ylabel('% values')

    img = (img - img.min()) / (img.max() - img.min())

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

    plt.subplot(2, 3, 6)
    plt.hist(phi, bins=45,
             weights=img.flatten() / np.sum(img) * 100)
    plt.ylim((0, 5))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('% values')

    plt.savefig('./benchmark_steerable.svg', dpi=300)

    plt.show()

    fig, axs = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(18, 5),
                            layout='constrained')

    for k, method in enumerate(methods):

        angles, aniso, *_ = method(images[2])
        if len(angles.shape) > 2:
            angles = angles[..., 0]
        if len(aniso.shape) > 2:
            aniso = aniso[..., 0]

        plt.subplot(2, 5, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow(angles[400:600, 400:600], cmap='twilight', clim=(0, 180))

        plt.subplot(2, 5, k + 6)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        data = np.clip((aniso - np.percentile(aniso, 1)) /
                       (np.percentile(aniso, 99) -
                        np.percentile(aniso, 1)), 0, 1)
        plt.imshow(data[400:600, 400:600], cmap='magma')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=axs[0], label='Angle (degrees)')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=axs[1],
                 label='Fiber-likeness score')

    plt.savefig('./benchmark_cross.svg', dpi=300)

    plt.show()

    fig, axs = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(18, 5),
                            layout='constrained')

    for k, method in enumerate(methods):

        angles, aniso, *_ = method(images[1])
        if len(angles.shape) > 2:
            angles = angles[..., 0]
        if len(aniso.shape) > 2:
            aniso = aniso[..., 0]

        plt.subplot(2, 5, k + 1)
        plt.title(f"({string.ascii_lowercase[k]})")
        if k == 0:
            plt.ylabel('1', rotation='horizontal')
        plt.imshow(angles, cmap='twilight', clim=(0, 180))

        plt.subplot(2, 5, k + 6)
        if k == 0:
            plt.ylabel('2', rotation='horizontal')
        data = np.clip((aniso - np.percentile(aniso, 1)) /
                       (np.percentile(aniso, 99) -
                        np.percentile(aniso, 1)), 0, 1)
        plt.imshow(data, cmap='magma')

    plt.colorbar(ScalarMappable(norm=Normalize(0, 180), cmap='twilight'),
                 ax=axs[0], label='Angle (degrees)')
    plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='magma'), ax=axs[1],
                 label='Fiber-likeness score')

    plt.savefig('./benchmark_lines.svg', dpi=300)

    plt.show()
