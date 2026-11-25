#!/usr/bin/env python3
"""Calculate the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    X: numpy.ndarray of shape (n, d) containing the data set
    pi: numpy.ndarray of shape (k,) containing the priors for each cluster
    m: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    S: numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster

    Returns: g, l, or None, None on failure
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
        l: the total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    k = pi.shape[0]
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.shape != (k, d):
        return None, None
    if not isinstance(S, np.ndarray) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None
    probs = np.zeros((k, X.shape[0]))
    for i in range(k):
        probs[i] = pi[i] * pdf(X, m[i], S[i])
    total = np.sum(probs, axis=0)
    return probs / total, np.sum(np.log(total))
