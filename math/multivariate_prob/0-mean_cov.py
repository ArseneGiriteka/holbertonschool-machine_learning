#!/usr/bin/env python3
"""Module that calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set

    X: numpy.ndarray of shape (n, d) containing the data set
    Returns: mean of shape (1, d) and cov of shape (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    cov = centered.T @ centered / (n - 1)
    return mean, cov
