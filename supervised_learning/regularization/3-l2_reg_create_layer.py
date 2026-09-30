#!/usr/bin/env python3
"""Create a Layer with L2 Regularization module"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer in TensorFlow that includes L2
    regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used
        lambtha: the L2 regularization parameter

    Returns:
        the output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer,
    )
    return layer(prev)
