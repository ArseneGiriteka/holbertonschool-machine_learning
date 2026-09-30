#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    if necessary, the image should be padded with 0's

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return convolved
