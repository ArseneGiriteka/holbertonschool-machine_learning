#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer, the number of clusters

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the priors for each
            cluster, initialized evenly
        m: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster, initialized with K-means
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    pi = np.full(k, 1 / k)
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
