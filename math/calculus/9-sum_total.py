#!/usr/bin/env python3
"""This module contains a function that compute
the sum of squares with the sum of squares formula"""


def summation_i_squared(n):
    """compute the sum of squares with the sum of squares
    formula"""
    if isinstance(n, (int)) and n >= 1:
        return int((n * (n + 1) * (2*n + 1)) / 6)
    else:
        return None
