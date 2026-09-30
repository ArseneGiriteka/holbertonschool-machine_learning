#!/usr/bin/env python3
"""Performs back propagation over a convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    padding is a string that is either same or valid
    stride is a tuple of (sh, sw) containing the strides for the convolution

    Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    A_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    dA_pad = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            x = i * sh
            y = j * sw
            for k in range(c_new):
                dz = dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]
                dA_pad[:, x:x + kh, y:y + kw, :] += dz * W[:, :, :, k]
                dW[:, :, :, k] += np.sum(
                    A_pad[:, x:x + kh, y:y + kw, :] * dz, axis=0)

    dA_prev = dA_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db
