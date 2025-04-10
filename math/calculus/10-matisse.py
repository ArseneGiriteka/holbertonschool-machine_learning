#!/usr/bin/env python3
"""this mudule compute the derivative of a polynomial
function"""


def poly_derivative(poly):
    """This function compute the derivative of a polynomial
    function"""

    if not isinstance(poly, list):
        return None
    elif len(poly) == 0:
        return None

    result = []
    length = len(poly)

    for i in range(length):
        if i != 0:
            result += [poly[i] * i]

    if length == 1:
        return [0]

    return result
