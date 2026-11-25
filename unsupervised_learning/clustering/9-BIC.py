#!/usr/bin/env python3
"""Find the best number of clusters for a GMM using the BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer, the minimum number of clusters (inclusive)
    kmax: positive integer, the maximum number of clusters (inclusive),
        set to the maximum number of clusters possible if None
    iterations: positive integer, the maximum number of iterations for
        the EM algorithm
    tol: non-negative float, the tolerance for the EM algorithm
    verbose: boolean, if True the EM algorithm prints information

    Returns: best_k, best_result, l, b, or None, None, None, None on
        failure
        best_k: the best value for k based on its BIC
        best_result: tuple containing pi, m, S for the best k
        l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    results = []
    lls = []
    bics = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, ll = expectation_maximization(X, k, iterations, tol,
                                                   verbose)
        if pi is None:
            return None, None, None, None
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        results.append((pi, m, S))
        lls.append(ll)
        bics.append(p * np.log(n) - 2 * ll)
    best = np.argmin(bics)
    return kmin + best, results[best], np.array(lls), np.array(bics)
