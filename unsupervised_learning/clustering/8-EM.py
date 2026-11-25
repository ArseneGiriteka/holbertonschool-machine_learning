#!/usr/bin/env python3
"""Perform the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Performs the expectation maximization for a GMM

    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer, the number of clusters
    iterations: positive integer, the maximum number of iterations
    tol: non-negative float, tolerance of the log likelihood used for
        early stopping
    verbose: boolean, if True prints the log likelihood every 10
        iterations and after the last iteration

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices
        g: numpy.ndarray of shape (k, n) containing the probabilities
            for each data point in each cluster
        l: the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    g, l_new = expectation(X, pi, m, S)
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(l_new, 5)))
        l_prev = l_new
        pi, m, S = maximization(X, g)
        g, l_new = expectation(X, pi, m, S)
        if abs(l_new - l_prev) <= tol:
            break
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i + 1, round(l_new, 5)))
    return pi, m, S, g, l_new
