#!/usr/bin/env python3
"""Makes a prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: boolean that determines if output should be printed
            during the prediction process

    Returns:
        the prediction for the data
    """
    return network.predict(data, verbose=verbose)
