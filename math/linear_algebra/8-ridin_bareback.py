#!/usr/bin/env python3
"""
This module contrains a function that multiply two 2D matrix
"""


def mat_mul(mat1, mat2):
    """
    This function multiply mat1 and mat2 2D matrix and return
    the result in a new list
    """
    result = []

    if len(mat1[0]) != len(mat2):
        return None

    for i in range(len(mat1)):
        result_row = []
        row = mat1[i]
        for j in range(len(mat2[0])):
            column = get_column(mat2, j)
            element = 0
            for k in range(len(row)):
                element += row[k] * column[k]
            result_row += [element]
            if j == len(mat2[0]) - 1:
                result += [list(result_row)]
    return result


def get_column(matrix, index):
    """
    This function returns the indexth column of given matrix
    """
    return [r[index] for r in matrix]
