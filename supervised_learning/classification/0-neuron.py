#!/usr/bin/env python3
"""Module that implements a neuron doing binary classification"""
import numpy as np


class Neuron:
    """implements a neuron doing binary classification"""

    def __init__(self, nx):
        """Initializes the neuron

        Args:
            nx (int): number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
