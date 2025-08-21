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
from microstructure_orientation.peak_detection import _find_peaks_gpu
from microstructure_orientation.gaussian_fit import _fit_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))

if __name__ == '__main__':

    img = np.load('/home/weis/Codes/MicrostructureOrientation/'
                  'artificial/cross.npy')
    filter_wavelength = 100
    sigma_x = 12
    sigma_y = 30

    img = (img - img.min()) / (img.max() - img.min())

    mem_pool = cp.get_default_memory_pool()

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
    var = np.sqrt(np.sum(np.power(res - np.mean(res, axis=-1)[..., np.newaxis],
                                  2), axis=-1))

    mem_pool.free_all_blocks()
    mem_pool = cp.get_default_memory_pool()

    angles, params = _find_peaks_gpu(res, np.linspace(0, 180, NB_ANGLES))

    mem_pool.free_all_blocks()
    mem_pool = cp.get_default_memory_pool()

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

    # Free up the GPU memory
    mem_pool.free_all_blocks()

    plt.rcParams['font.family'] = 'serif'

    plt.figure()
    plt.imshow(img, cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/image.svg', dpi=300)

    plt.figure()
    plt.imshow(img[450:550, 450:550], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Pixel value')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/image_close.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 0], cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[450:550, 450:550, 0], cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/dominant_angle_close.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 1], cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/second_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[450:550, 450:550, 1], cmap='twilight')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Angle (degrees)')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/second_angle_close.svg', dpi=300)

    plt.figure()
    plt.imshow(param[..., 0], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Standard deviation')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/std.svg', dpi=300)

    plt.figure()
    plt.imshow(param[450:550, 450:550, 0], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Standard deviation')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/std_close.svg', dpi=300)

    plt.figure()
    plt.imshow(param[..., 1] / param[..., 0], cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Amplitude / std')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/intensity_indicator.svg', dpi=300)

    plt.figure()
    plt.imshow(param[450:550, 450:550, 1] / param[450:550, 450:550, 0],
               cmap='magma')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Amplitude / std')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/intensity_indicator_close.svg', dpi=300)

    plt.figure()
    plt.hist(angles[..., 0].flatten(),
             weights=np.full_like(angles[..., 0].flatten(),
                                  1 / (angles.shape[0] * angles.shape[1])),
             bins=45)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Fraction of values')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'artificial/cross/angle_histogram.svg', dpi=300)

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title('A')
    plt.ylim((0.5, 1.0))
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[511, 534])
    plt.vlines(angles[511, 534, 0],
               res[511, 534, int(round(angles[511, 534, 0] * 44 / 180, 0))]
               - 0.07,
               res[511, 534, int(round(angles[511, 534, 0] * 44 / 180, 0))]
               + 0.07,
               color='k', alpha=0.5)
    plt.subplot(132)
    plt.title('B')
    plt.ylim((0.5, 1.0))
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
    plt.ylim((0.5, 1.0))
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
                'artificial/cross/angular_distributions.svg', dpi=300)

    plt.close('all')
