#!/usr/bin/env python3
"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for
        the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    pooled = np.zeros((m, output_h, output_w, c))

    if mode == 'max':
        op = np.max
    else:
        op = np.average

    for i in range(output_h):
        for j in range(output_w):
            x = i * sh
            y = j * sw
            pooled[:, i, j, :] = op(
                images[:, x:x + kh, y:y + kw, :], axis=(1, 2))

    return pooled
