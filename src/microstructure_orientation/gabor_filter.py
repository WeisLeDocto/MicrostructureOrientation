# coding: utf-8

from pathlib import Path
from time import time
import numpy as np
import cucim.skimage.filters as gpu_filters
import cupy as cp
import cupyx.scipy.signal as gpu_signal
from tqdm.auto import tqdm
import sys
import os

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def _process_gabor_gpu(image: np.ndarray,
                       kernels: dict[int, cp.ndarray],
                       nb_angles: int) -> np.ndarray:
    """Performs a single filtering step of an image by multiple Gabor kernel on
    the GPU.

    Args:
        image: The exposure fusion image on which to perform the filtering.
        kernels: A dictionary containing for each orientation the associated
            Gabor filter, in GPU memory.
        nb_angles: Number of successive angles to use for the filters.

    Returns:
        Result numpy array containing for each orientation the intensity of the
        response to the Gabor filters.
    """

    # Load input image on GPU and prepare the output array
    img_gpu = cp.asarray(image, dtype='float32')
    res = cp.zeros(shape=(*image.shape, nb_angles), dtype='float32')

    # Apply Gabor filters on all the images
    for i, kernel in tqdm(kernels.items(),
                          total=nb_angles,
                          desc='Apply Gabor kernels',
                          file=sys.stdout,
                          colour='green',
                          mininterval=0.001,
                          maxinterval=0.01,
                          position=1,
                          leave=False):
        conv = gpu_signal.convolve2d(img_gpu,
                                     kernel,
                                     mode='same',
                                     boundary='symm').astype(cp.complex64)
        # Normalize by the norm of the filter
        conv /= cp.linalg.norm(cp.sum(kernel))
        res[:, :, i] = cp.sqrt(conv.real ** 2 + conv.imag ** 2)

    # Return a numpy array for saving on disk
    return cp.asnumpy(res)


def apply_gabor_filter(src_path: Path,
                       dest_path: Path,
                       filter_wavelength: int) -> None:
    """Applies a Gabor filter twice on exposure fusion images, to detect the
    orientation of fibers.

    Args:
        src_path: Path to the folder containing the exposure fusion images.
        dest_path: Path to the folder where to write the filtered images.
        filter_wavelength: Spatial wavelength to use for the Gabor filter,
            roughly equal to the width of the fibers to detect.
    """

    # Create the folder containing the output arrays
    dest_path.mkdir(parents=False, exist_ok=True)

    # Load all the convolution kernels at once in the GPU
    print('\nSetting convolution kernels')
    t0 = time()
    kernels = {i: gpu_filters.gabor_kernel(frequency=1 / filter_wavelength,
                                           theta=np.pi / 2 - ang,
                                           n_stds=2,
                                           offset=0,
                                           bandwidth=1,
                                           dtype=cp.complex64,
                                           sigma_x=4,
                                           sigma_y=7.5)
               for i, ang
               in enumerate(np.linspace(0, np.pi, NB_ANGLES).tolist())}
    print(f'Set convolution kernels in {time() - t0:.2f}s\n')

    # Apply on each image all the Gabor filters defined above
    images = tuple(sorted(src_path.glob('*.npy')))
    for img_path in tqdm(images,
                         total=len(images),
                         desc='Applying Gabor filter',
                         file=sys.stdout,
                         colour='green',
                         mininterval=0.001,
                         maxinterval=0.01,
                         position=0,
                         leave=True):
        # Load the image, normalize it and invert the intensity
        img_path: Path
        img = np.load(img_path)
        img = 1.0 - (img - img.min()) / (img.max() - img.min())

        # Apply the Gabor filters and save the resulting array
        res = _process_gabor_gpu(img, kernels, NB_ANGLES)
        np.save(dest_path / img_path.name, res)

    # Free up some GPU memory as the kernels are no longer needed
    del kernels
