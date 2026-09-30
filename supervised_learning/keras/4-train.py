#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient
            descent
        verbose: boolean that determines if output should be printed
            during training
        shuffle: boolean that determines whether to shuffle the batches
            every epoch

    Returns:
        the History object generated after training the model
    """
    history = network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose, shuffle=shuffle)

    return history
