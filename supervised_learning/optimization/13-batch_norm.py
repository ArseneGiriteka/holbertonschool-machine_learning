#!/usr/bin/env python3
"""Normalizes an unactivated output of a neural network using
batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using
    batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing the scales
            used for batch normalization
        beta: numpy.ndarray of shape (1, n) containing the offsets
            used for batch normalization
        epsilon: a small number used to avoid division by zero

    Returns:
        the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
