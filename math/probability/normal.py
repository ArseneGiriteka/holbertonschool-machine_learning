#!/usr/bin/env python3
"""Module that defines the Normal distribution class"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the Normal distribution

        data: list of data used to estimate the distribution
        mean: mean of the distribution
        stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total = 0
            for x in data:
                total += x
            self.mean = float(total / len(data))
            squares = 0
            for x in data:
                squares += (x - self.mean) ** 2
            self.stddev = float((squares / len(data)) ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculates the PDF value for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return e ** exponent / (self.stddev * (2 * pi) ** 0.5)
