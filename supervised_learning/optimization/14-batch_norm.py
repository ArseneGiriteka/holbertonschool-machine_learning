#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network
    in tensorflow

    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function that should be used on
            the output of the layer

    Returns:
        a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.ones((1, n)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((1, n)), trainable=True)
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon)

    return activation(Z_norm)
