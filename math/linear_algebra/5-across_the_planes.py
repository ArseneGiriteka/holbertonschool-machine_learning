#!/usr/bin/env python3
"""
this module contains:
- add_matrices: to add 2D matrices
"""


def add_matrices2D(mat1, mat2):
    """
    This function computes addition over two 2D matrix:
    all element at the same position have same dimmension
    accepts: mat1, mat2 -> 2D matrix(int|float)
    return: - if dim(mat1) == dim(mat2) returns new matrix
            - if dim(mat1) != dim(mat2) returns None
    """
    if len(mat1) != len(mat2) or\
            len(mat2[0]) != len(mat1[0]):
        return None

    sum = []
    for i in range(len(mat2)):
        sum += [[mat1[i][j] + mat2[i][j] for j in range(len(mat2[0]))]]

    return sum
