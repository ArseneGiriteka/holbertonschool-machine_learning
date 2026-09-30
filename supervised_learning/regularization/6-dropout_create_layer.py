#!/usr/bin/env python3
"""Create a Layer with Dropout module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function for the new layer
        keep_prob: the probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        the output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
    )
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(dense(prev), training=training)
