#!/usr/bin/env python3
"""Defines a function that calculates specificity for each class"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    total = np.sum(confusion)
    true_negatives = total - true_positives - false_positives - \
        false_negatives
    return true_negatives / (true_negatives + false_positives)
