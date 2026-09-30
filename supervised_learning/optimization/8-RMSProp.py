#!/usr/bin/env python3
"""Sets up the RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Sets up the RMSProp optimization algorithm in TensorFlow

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight (Discounting factor)
        epsilon: a small number to avoid division by zero

    Returns:
        optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
