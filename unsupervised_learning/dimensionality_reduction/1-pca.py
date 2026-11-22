#!/usr/bin/env python3
"""Principal Component Analysis (PCA) to a fixed dimensionality"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset

    X: numpy.ndarray of shape (n, d) containing the dataset
    ndim: new dimensionality of the transformed X

    Returns: T, numpy.ndarray of shape (n, ndim) containing the
    transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_m)
    return np.matmul(X_m, vh[:ndim].T)
