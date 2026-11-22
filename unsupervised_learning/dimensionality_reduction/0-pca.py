#!/usr/bin/env python3
"""Principal Component Analysis (PCA) on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    X: numpy.ndarray of shape (n, d), every dimension has a mean of 0
    var: fraction of the variance the transformation should keep

    Returns: the weights matrix W of shape (d, nd), nd being the new
    dimensionality that keeps var fraction of X's original variance
    """
    _, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    nd = np.argwhere(cum_var >= var)[0, 0] + 1
    return vh[:nd].T
