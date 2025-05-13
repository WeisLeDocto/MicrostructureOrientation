# coding: utf-8

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def image_correlation(ref_img: np.ndarray,
                      def_img: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs digital image correlation on the two provided images, using
    cv2's DISFlow, and computes the Hencky strain.

    Args:
        ref_img: The reference image for correlation.
        def_img: The deformed image.

    Returns:
        The xx, yy, and xy components of the Hencky strain, as numpy arrays.
    """

    ref_img_cont = ((ref_img - ref_img.min() + 1) /
                    (ref_img.max() - ref_img.min() + 1) * 255).astype(np.uint8)
    def_img_cont = ((def_img - def_img.min() + 1) /
                    (def_img.max() - def_img.min() + 1) * 255).astype(np.uint8)

    # Configure DISFlow for our specific application
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementAlpha(15.0)
    dis.setVariationalRefinementDelta(1.0)
    dis.setVariationalRefinementEpsilon(0.01)
    dis.setVariationalRefinementGamma(0.0)
    dis.setFinestScale(0)
    dis.setVariationalRefinementIterations(100)
    dis.setGradientDescentIterations(25)
    dis.setPatchSize(8)
    dis.setPatchStride(1)
    dis.setUseMeanNormalization(True)

    # Compute the optical flow between the two provided images
    flow = dis.calc(np.ascontiguousarray(ref_img_cont),
                    np.ascontiguousarray(def_img_cont), None)

    # Apply a Gaussian filter to remove noise
    flow = gaussian_filter(flow, 1, order=0)

    # Get the gradient from the optical flow
    du_dx = np.gradient(flow[:, :, 0], axis=1)
    du_dy = np.gradient(flow[:, :, 0], axis=0)
    dv_dx = np.gradient(flow[:, :, 1], axis=1)
    dv_dy = np.gradient(flow[:, :, 1], axis=0)

    # Compute the deformation gradient from the gradient
    du_dx += 1.0
    dv_dy += 1.0
    def_grad = np.stack((np.stack((du_dx, dv_dx), axis=2),
                         np.stack((du_dy, dv_dy), axis=2)), axis=3)

    # Compute the Hencky strain tensor from the deformation gradient
    # We use diagonalize as matrix log cannot be easily vectorized
    cauchy = np.transpose(def_grad, axes=(0, 1, 3, 2)) @ def_grad
    vals, vects = np.linalg.eigh(cauchy)
    vals = np.nan_to_num(np.log(vals))
    vals = np.stack((np.stack((vals[:, :, 0], np.zeros_like(vals[:, :, 0])),
                              axis=2),
                     np.stack((np.zeros_like(vals[:, :, 1]), vals[:, :, 1]),
                              axis=2)), axis=3)
    hencky = 0.5 * (vects @ vals @ np.linalg.inv(vects))

    return hencky[:, :, 0, 0], hencky[:, :, 1, 1], hencky[:, :, 0, 1]
