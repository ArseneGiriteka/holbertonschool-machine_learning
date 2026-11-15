#!/usr/bin/env python3
"""Module that calculates the cofactor matrix of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a square matrix

    Args:
        matrix: non-empty square list of lists

    Returns:
        the determinant of matrix
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(n):
        sub = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix

    Args:
        matrix: list of lists whose cofactor matrix should be calculated

    Returns:
        the cofactor matrix of matrix
    """
    if (not isinstance(matrix, list) or len(matrix) == 0 or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    return [[((-1) ** (i + j)) *
             determinant([r[:j] + r[j + 1:]
                          for k, r in enumerate(matrix) if k != i])
             for j in range(n)] for i in range(n)]
