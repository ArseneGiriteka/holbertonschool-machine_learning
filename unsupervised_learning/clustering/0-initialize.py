#!/usr/bin/env python3
"""Initialize cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer, the number of clusters

    Returns: numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    return np.random.uniform(X.min(axis=0), X.max(axis=0),
                             size=(k, X.shape[1]))
