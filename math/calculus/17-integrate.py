#!/usr/bin/env python3
"""function def poly_integral(poly, C=0)
that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function def poly_integral(poly, C=0)
    that calculates the integral of a polynomial"""
    if not isinstance(poly, list):
        return None
    elif len(poly) == 0:
        return None

    if not isinstance(C, (int)):
        return None

    result = [C]
    length = len(poly)

    if poly == [0]:
        return result

    for i in range(length):
        coeff = poly[i] / (i + 1)

        if coeff == int(coeff):
            coeff = int(coeff)
        result += [coeff]

    return result
