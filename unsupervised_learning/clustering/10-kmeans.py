#!/usr/bin/env python3
"""Perform K-means on a dataset with scikit-learn"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: the number of clusters

    Returns: C, clss
        C: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        clss: numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
