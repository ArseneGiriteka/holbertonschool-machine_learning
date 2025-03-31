#!/usr/bin/env python3
"""
This module contains a function that return a transpose of a given
non empty, same type/shape elements.
"""


def matrix_transpose(matrix):
    """
    this function accepts a matrix and returns its
    transpose
    """
    matrix_copy = matrix
    transpose = []

    for i in range(len(matrix_copy[0])):
        transpose += [column(matrix_copy, i)]

    return transpose


def column(matrix, index):
    """
    This function returns the indexth column of given matrix
    """
    return [r[index] for r in matrix]
