#!/usr/bin/env python3
"""Module that defines a deep neural network
performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initializes the deep neural network

        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            nodes = layers[i]
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
            prev = nx if i == 0 else layers[i - 1]
            self.__weights["W" + str(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev))
            self.__weights["b" + str(i + 1)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights
