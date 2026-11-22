# Bayesian Probability

This project covers the basics of Bayesian probability with NumPy:
computing the likelihood of observed data, the intersection and marginal
probabilities, and the posterior probability using Bayes' rule.

## Learning Objectives

- What is Bayesian Probability?
- What is Bayes' rule and how do you use it?
- What is a base rate?
- What is a prior?
- What is a posterior?
- What is a likelihood?

## Requirements

- Ubuntu 20.04 LTS, `python3` (version 3.9)
- `numpy` (version 1.25.2), imported as `import numpy as np`
- `pycodestyle` (version 2.11.1)

## Tasks

| File | Description |
| --- | --- |
| `0-likelihood.py` | `likelihood(x, n, P)` - likelihood of obtaining `x` out of `n` for each hypothetical probability in `P` (binomial distribution) |
| `1-intersection.py` | `intersection(x, n, P, Pr)` - intersection of obtaining `x` out of `n` with each probability in `P`, given the prior beliefs `Pr` |
| `2-marginal.py` | `marginal(x, n, P, Pr)` - marginal probability of obtaining `x` out of `n` (sum of the intersections over every probability in `P`) |
| `3-posterior.py` | `posterior(x, n, P, Pr)` - posterior probability of each probability in `P` given `x` and `n` (intersection / marginal, Bayes' rule) |
