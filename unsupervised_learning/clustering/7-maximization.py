#!/usr/bin/env python3
"""Calculate the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    X: numpy.ndarray of shape (n, d) containing the data set
    g: numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the updated priors
            for each cluster
        m: numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    if g.shape[1] != n:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None
    k = g.shape[0]
    nk = np.sum(g, axis=1)
    pi = nk / n
    m = g @ X / nk[:, None]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i, :, None] * diff).T @ diff / nk[i]
    return pi, m, S
