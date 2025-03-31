#!/usr/bin/env python3
"""
This module contains:
- matrix_shape(matrix): a function that compute a shape of a
given matrix
"""


def matrix_shape(matrix):
    """
    This function computesthe shave of matrix a given
    matrix with all elements in the same dimension are of
    the same type/shape

    The shape is returned as a list of integers
    """

    size = []
    while isinstance(matrix, list):
        size.append(len(matrix))
        if not isinstance(matrix[0], list):
            break
        else:
            matrix = matrix[0]

    return size
