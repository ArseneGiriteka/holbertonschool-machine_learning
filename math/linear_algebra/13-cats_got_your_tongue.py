#!/usr/bin/env python3
"""
This module contains a function that concantenate numpy.ndarrays
along different axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function concantenate mat1 and mat2 numpy.ndarrays on a
    specific axis
    """
    return np.concatenate((mat1, mat2), axis=axis)
