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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from microstructure_orientation.peak_detection import _find_peaks_gpu
from microstructure_orientation.gaussian_fit import _fit_gpu

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def periodic_gauss(x: np.ndarray,
                   sigma_1: float,
                   a_1: float,
                   sigma_2: float | None,
                   a_2: float | None,
                   sigma_3: float | None,
                   a_3: float | None,
                   b: float,
                   mu_1: float,
                   mu_2: float | None,
                   mu_3: float | None,
                   n: int) -> np.ndarray:
    """Computes the sum of three periodic gaussian curves of the input array,
    using the provided gaussian parameters.

    Args:
        x: The array over which to compute the gaussians.
        sigma_1: Standard deviation for the first gaussian.
        a_1: Multiplicative factor for the first gaussian.
        sigma_2: Standard deviation for the second gaussian.
        a_2: Multiplicative factor for the second gaussian.
        sigma_3: Standard deviation for the second gaussian.
        a_3: Multiplicative factor for the second gaussian.
        b: Offset, common to all the gaussians.
        mu_1: Center value of the first gaussian.
        mu_2: Center value of the second gaussian.
        mu_3: Center value of the third gaussian.
        n: Number of gaussians to compute, between 1 and 3.

    Returns:
        The sum of the three periodic gaussians over the input array.
    """

    array_out = np.empty_like(x)

    if n == 1:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)))
    elif n == 2:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)) +
                a_2 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_2,
                                                    math.pi) -
                                          math.pi / 2) / sigma_2, 2)))
    elif n == 3:
        for i in range(x.shape[0]):
            array_out[i] = (
                b +
                a_1 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_1,
                                                    math.pi) -
                                          math.pi / 2) / sigma_1, 2)) +
                a_2 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_2,
                                                    math.pi) -
                                          math.pi / 2) / sigma_2, 2)) +
                a_3 * math.exp(-math.pow((math.fmod(x[i] + math.pi / 2 - mu_3,
                                                    math.pi) -
                                          math.pi / 2) / sigma_3, 2)))
    return array_out


if __name__ == '__main__':

    # img = np.load('/home/weis/Desktop/HDR/7LX1_2/hdr/0_1714.npy')
    # img = np.load('/home/weis/Desktop/HDR/7LX1/hdr/2_3260.npy')
    img = np.load('/home/weis/Desktop/HDR/7LX1_3/hdr/7_04362.npy')
    filter_wavelength = 100
    sigma_x = 4
    sigma_y = 10

    img = 1.0 - (img - img.min()) / (img.max() - img.min())

    plt.rcParams['font.family'] = 'serif'

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/image.svg', dpi=300)

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

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cp.asnumpy(cp.real(kernels[0])), cmap='plasma')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/kernel_0.svg', dpi=300)

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cp.asnumpy(cp.real(kernels[22])), cmap='plasma')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/kernel_90.svg', dpi=300)

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

    plt.figure()
    plt.imshow(res[..., 0], cmap='plasma')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/gabor_0.svg', dpi=300)

    plt.figure()
    plt.imshow(res[..., 22], cmap='plasma')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/gabor_90.svg', dpi=300)

    plt.figure()
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[215, 591])
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/angle_distribution.svg', dpi=300)

    mem_pool.free_all_blocks()
    mem_pool = cp.get_default_memory_pool()

    angles, params = _find_peaks_gpu(res, np.linspace(0, 180, NB_ANGLES))

    plt.figure()
    plt.imshow(angles[..., 0], cmap='twilight')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/dominant_angle.svg', dpi=300)

    plt.figure()
    plt.imshow(angles[..., 1], cmap='twilight')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/second_angle.svg', dpi=300)

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

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.linspace(0, 180, NB_ANGLES), res[215, 591])
    plt.plot(np.linspace(0, 180, NB_ANGLES),
             periodic_gauss(np.linspace(0, np.pi, NB_ANGLES),
                            *param[215, 591],
                            *np.radians(angles[215, 591]),
                            n_peaks[215, 591]), color='k')

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/gaussian_fit.svg', dpi=300)

    plt.figure()
    plt.imshow(param[..., 0], cmap='plasma')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/std.svg', dpi=300)

    plt.figure()
    plt.imshow(param[..., 1], cmap='plasma',
               clim=(np.percentile(param[..., 1], 1),
                     np.percentile(param[..., 1], 99)))
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/amplitude.svg', dpi=300)

    plt.figure()
    data = param[..., 1] / param[..., 0]
    plt.imshow(data, cmap='plasma', clim=(np.percentile(data, 1),
                                          np.percentile(data, 99)))
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/weis/Codes/MicrostructureOrientation/'
                'figures/graphical_abstract/anisotropy.svg', dpi=300)

    plt.show()
