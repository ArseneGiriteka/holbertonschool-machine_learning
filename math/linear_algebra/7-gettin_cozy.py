#!/usr/bin/env python3
"""
this module contains a function that concantenates two matrix
along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    this function takes two 2D matrix and an axis (by default 0)
    to concantanate on after it returns a new matrix
    """
    if axis == 0:
        if len(mat2[0]) != len(mat1[0]):
            return None
        return [list(mr1) for mr1 in mat1] + [list(mr2) for mr2 in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [list(m1) + list(m2) for m1, m2 in zip(mat1, mat2)]
    else:
        return None
