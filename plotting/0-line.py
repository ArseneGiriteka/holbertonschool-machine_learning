#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
this module contains a function that plot a
red graph
"""


def line():
    """
    plot a red solid graph with x range [0, 10]
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color="red", linestyle="-")
    plt.xlim(0, 10)
    plt.show()
