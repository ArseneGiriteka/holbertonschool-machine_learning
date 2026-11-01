#!/usr/bin/env python3
"""Module that defines the Binomial distribution class"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the Binomial distribution

        data: list of data used to estimate the distribution
        n: number of Bernoulli trials
        p: probability of a success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total = 0
            for x in data:
                total += x
            mean = total / len(data)
            squares = 0
            for x in data:
                squares += (x - mean) ** 2
            variance = squares / len(data)
            p = 1 - variance / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the PMF value for a given number of successes k"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(m):
            """Returns the factorial of m"""
            result = 1
            for i in range(2, m + 1):
                result *= i
            return result

        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the CDF value for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            k = self.n
        fact = [1]
        for i in range(1, self.n + 1):
            fact.append(fact[-1] * i)
        q = 1 - self.p
        total = 0
        for i in range(k + 1):
            total += fact[self.n] * (self.p ** i) * (q ** (self.n - i)) / \
                (fact[i] * fact[self.n - i])
        return total
