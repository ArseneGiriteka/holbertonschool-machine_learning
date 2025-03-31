#!/usr/bin/env python3
"""
This module contains a function that compute element-wise
addition, substration, multiplication, division operation
and return a tuple(sum, difference, production, quotient)
"""


def np_elementwise(mat1, mat2):
    """
    THis function compute element-wise operations(addition,
    substraction, division, quotient) over mat1, mat2
    numpy.ndarray obects and return a tuple(addition,
    substraction, division, quotient) as a result
    """
    return tuple((mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2))
