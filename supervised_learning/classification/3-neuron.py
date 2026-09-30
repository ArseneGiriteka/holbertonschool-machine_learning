#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initializes the neuron

        Args:
            nx (int): number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): input data of shape (nx, m)

        Returns:
            The activated output of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): correct labels of shape (1, m)
            A (numpy.ndarray): activated output of shape (1, m)

        Returns:
            The cost
        """
        m = Y.shape[1]
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return -np.sum(loss) / m
