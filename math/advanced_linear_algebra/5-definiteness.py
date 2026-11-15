#!/usr/bin/env python3
"""Module that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix

    Args:
        matrix: numpy.ndarray of shape (n, n) whose definiteness
            should be calculated

    Returns:
        Positive definite, Positive semi-definite, Negative semi-definite,
        Negative definite or Indefinite, or None if matrix is not a valid
        matrix or does not fit any of these categories
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if (matrix.ndim != 2 or matrix.shape[0] == 0 or
            matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.allclose(matrix, matrix.T):
        return None
    eig = np.linalg.eigvalsh(matrix)
    tol = 1e-8 * max(1.0, np.max(np.abs(eig)))
    pos = np.sum(eig > tol)
    neg = np.sum(eig < -tol)
    n = len(eig)
    if pos == n:
        return "Positive definite"
    if neg == n:
        return "Negative definite"
    if neg == 0:
        return "Positive semi-definite"
    if pos == 0:
        return "Negative semi-definite"
    return "Indefinite"
