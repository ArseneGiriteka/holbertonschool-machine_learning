#!/usr/bin/env python3
"""Performs forward propagation over a pooling layer"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
    stride is a tuple of (sh, sw) containing the strides for the pooling
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1
    A = np.zeros((m, h_new, w_new, c_prev))

    pool = np.max if mode == 'max' else np.mean

    for i in range(h_new):
        for j in range(w_new):
            x = i * sh
            y = j * sw
            A[:, i, j, :] = pool(
                A_prev[:, x:x + kh, y:y + kw, :], axis=(1, 2))

    return A
