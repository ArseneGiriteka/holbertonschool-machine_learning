#!/usr/bin/env python3
import numpy as np
"""
This module contains a function that return a transpose of a given
non empty, same type/shape elements.
"""


def matrix_transpose(matrix):
    """
    this function accepts a matrix and returns its
    transpose
    """
    matrix_copy = np.array(matrix)

    return matrix_copy.T.tolist()
