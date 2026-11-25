#!/usr/bin/env python3
"""Perform agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset
    and displays the dendrogram with each cluster in a different color

    X: numpy.ndarray of shape (n, d) containing the dataset
    dist: the maximum cophenetic distance for all clusters

    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
        indices for each data point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
