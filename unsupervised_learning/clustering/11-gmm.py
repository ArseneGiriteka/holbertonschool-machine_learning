#!/usr/bin/env python3
"""Calculate a GMM from a dataset with scikit-learn"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: the number of clusters

    Returns: pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,) containing the cluster priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices
        clss: numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
        bic: the BIC value of the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
