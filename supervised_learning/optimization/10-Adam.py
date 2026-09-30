#!/usr/bin/env python3
"""Sets up the Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Sets up the Adam optimization algorithm in TensorFlow

    Args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero

    Returns:
        optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
