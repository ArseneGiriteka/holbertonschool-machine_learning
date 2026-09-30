#!/usr/bin/env python3
"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        the keras model
    """
    regularizer = K.regularizers.L2(lambtha)
    inputs = K.Input(shape=(nx,))
    outputs = inputs

    for i in range(len(layers)):
        outputs = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=regularizer)(outputs)
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)

    model = K.Model(inputs=inputs, outputs=outputs)

    return model
