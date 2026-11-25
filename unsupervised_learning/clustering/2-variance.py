#!/usr/bin/env python3
"""Calculate the total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    X: numpy.ndarray of shape (n, d) containing the data set
    C: numpy.ndarray of shape (k, d) containing the centroid means

    Returns: var, the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    dist = np.sum((X[:, None] - C) ** 2, axis=2)
    return np.sum(np.min(dist, axis=1))
