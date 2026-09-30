#!/usr/bin/env python3
"""Defines a function that calculates sensitivity for each class"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
