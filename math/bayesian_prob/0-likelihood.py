#!/usr/bin/env python3
"""Likelihood of binomial data for various hypothetical probabilities"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining x successes out of n trials

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities

    Returns:
        1D numpy.ndarray with the likelihood of the data for each
        probability in P
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
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    comb = 1
    for i in range(1, min(x, n - x) + 1):
        comb = comb * (n - i + 1) // i

    return comb * (P ** x) * ((1 - P) ** (n - x))
