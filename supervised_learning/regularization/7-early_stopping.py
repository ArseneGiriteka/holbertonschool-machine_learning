#!/usr/bin/env python3
"""Early Stopping module"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early

    Args:
        cost: the current validation cost of the neural network
        opt_cost: the lowest recorded validation cost of the network
        threshold: the threshold used for early stopping
        patience: the patience count used for early stopping
        count: the count of how long the threshold has not been met

    Returns:
        a boolean of whether the network should be stopped early,
        followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count >= patience, count
