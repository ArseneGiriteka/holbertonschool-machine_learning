# Multivariate Probability

This project covers the basics of multivariate probability with NumPy:
computing the mean and covariance of a data set, turning a covariance
matrix into a correlation matrix, and modelling data with a multivariate
normal (Gaussian) distribution.

## Learning Objectives

- Who is Carl Friedrich Gauss?
- What is a joint/multivariate distribution?
- What is a covariance?
- What is a correlation coefficient?
- What is a covariance matrix?
- What is a multivariate Gaussian distribution?

## Requirements

- Ubuntu 20.04 LTS, `python3` (version 3.9)
- `numpy` (version 1.25.2), imported as `import numpy as np`
- Code follows `pycodestyle` (version 2.11.1)
- Every file starts with `#!/usr/bin/env python3`, ends with a new line
  and is executable
- Every module, class and function is documented
- `numpy.cov` is not allowed

## Tasks

| # | File | Description |
|---|------|-------------|
| 0 | [`0-mean_cov.py`](0-mean_cov.py) | `mean_cov(X)` returns the mean `(1, d)` and covariance matrix `(d, d)` of a data set `X` of shape `(n, d)` |
| 1 | [`1-correlation.py`](1-correlation.py) | `correlation(C)` returns the correlation matrix of a covariance matrix `C` of shape `(d, d)` |
| 2 | [`multinormal.py`](multinormal.py) | `MultiNormal(data)` class: stores the mean `(d, 1)` and covariance `(d, d)` of a data set of shape `(d, n)` |
| 3 | [`multinormal.py`](multinormal.py) | `MultiNormal.pdf(x)` returns the probability density at a data point `x` of shape `(d, 1)` |

## Key Formulas

- Covariance matrix: `Σ = (X - μ)ᵀ (X - μ) / (n - 1)`
- Correlation: `ρᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)`
- Multivariate normal PDF:
  `f(x) = exp(-½ (x - μ)ᵀ Σ⁻¹ (x - μ)) / √((2π)ᵈ |Σ|)`

## Author

Arsène Giriteka
