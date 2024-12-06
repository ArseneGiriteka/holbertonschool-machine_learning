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
    if not mat1 or not mat2:
        return None

    if len(mat1)== 0 or len(mat2) == 0:
        return None
    
    if not mat1[0] or not mat2[0]:
        return

    if len(mat1) != len(mat2) or\
            len(mat2[0]) != len(mat1[0]):
        return None

    sum = []
    for i in range(len(mat2)):
        sum += [sum_row(mat1, mat2, i)]

    return sum

def sum_row(mat1, mat2, index):
    """
    This function returns the sum of indexth rows of mat1 and mat2
    mat1 and mat2 have same length
    """
    mat1_row = mat1[index]
    mat2_row = mat2[index]
    row_sum = []

    for i in range(len(mat1)):
        row_sum += [mat1_row[i] + mat2_row[i]]

    return row_sum
