#!/usr/bin/env python3
"""
This module add two arrays element-wise:
- arrays arguments are type int/float
- if arrays have different shapes None is returned
- a new list is returned
"""


def add_arrays(arr1, arr2):
    """
    This method adds two arrays element-wise
    """

    if len(arr1) != len(arr2):
        return None

    sum = []
    for i in range(len(arr1)):
        sum += [arr1[i] + arr2[i]]

    return sum
