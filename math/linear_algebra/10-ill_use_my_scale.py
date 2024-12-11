#!/usr/bin/env python3
"""
This module contains a function that calculate a shape of
a numpy.ndarray
"""
import numpy as np


def np_shape(matrix: np.ndarray):
    """
    this function takes a numpy.ndarray matrix and
    calculates its shape that will be returned as a
    tuple of ints
    """
    return matrix.shape
