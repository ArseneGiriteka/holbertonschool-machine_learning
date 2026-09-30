#!/usr/bin/env python3
"""Performs back propagation over a pooling layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    dA is a numpy.ndarray of shape (m, h_new, w_new, c) containing the
        partial derivatives with respect to the output of the pooling layer
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
        the output of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
    stride is a tuple of (sh, sw) containing the strides for the pooling
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the previous layer
        (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(h_new):
        for j in range(w_new):
            x = i * sh
            y = j * sw
            da = dA[:, i, j, np.newaxis, np.newaxis, :]
            if mode == 'max':
                window = A_prev[:, x:x + kh, y:y + kw, :]
                mask = window == np.max(window, axis=(1, 2), keepdims=True)
                dA_prev[:, x:x + kh, y:y + kw, :] += mask * da
            else:
                dA_prev[:, x:x + kh, y:y + kw, :] += da / (kh * kw)

    return dA_prev
