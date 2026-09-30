#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set

    Args:
        data: the list of data to calculate the moving average of
        beta: the weight used for the moving average

    Returns:
        a list containing the moving averages of data
    """
    moving_averages = []
    v = 0
    for i, value in enumerate(data):
        v = beta * v + (1 - beta) * value
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
