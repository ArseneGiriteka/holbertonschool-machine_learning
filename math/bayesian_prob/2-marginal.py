#!/usr/bin/env python3
"""Marginal probability of obtaining binomial data given prior beliefs"""
import numpy as np


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining x out of n,
    summed over every hypothetical probability in P and its prior

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of the prior beliefs of P

    Returns:
        the marginal probability of obtaining x and n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    comb = 1
    for i in range(1, min(x, n - x) + 1):
        comb = comb * (n - i + 1) // i

    return np.sum(comb * (P ** x) * ((1 - P) ** (n - x)) * Pr)
