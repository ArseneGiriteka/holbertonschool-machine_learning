#!/usr/bin/env python3
"""Calculate the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    X: numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m: numpy.ndarray of shape (d,) containing the mean of the distribution
    S: numpy.ndarray of shape (d, d) containing the covariance of the
        distribution

    Returns: P, or None on failure
        P: numpy.ndarray of shape (n,) containing the PDF values for each
            data point, with a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None
    diff = X - m
    expo = np.sum(diff @ np.linalg.inv(S) * diff, axis=1)
    norm = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    P = np.exp(-expo / 2) / norm
    return np.maximum(P, 1e-300)
