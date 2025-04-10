#!/usr/bin/env python3
"""
This module contains a function that draws
a line
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    This function draws a red culve line
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
