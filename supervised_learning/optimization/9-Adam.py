#!/usr/bin/env python3
"""Updates a variable using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment of var
        s: the previous second moment of var
        t: the time step used for bias correction

    Returns:
        the updated variable, the new first moment, and the new second
        moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
