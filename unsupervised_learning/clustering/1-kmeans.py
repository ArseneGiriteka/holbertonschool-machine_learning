#!/usr/bin/env python3
"""Perform K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer, the number of clusters
    iterations: positive integer, the maximum number of iterations

    Returns: C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))
    for _ in range(iterations):
        prev = C.copy()
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
        for j in range(k):
            points = X[clss == j]
            if points.shape[0] == 0:
                C[j] = np.random.uniform(low, high)
            else:
                C[j] = points.mean(axis=0)
        if np.array_equal(C, prev):
            return C, clss
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
    return C, clss
