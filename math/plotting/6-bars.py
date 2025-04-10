#!/usr/bin/env python3
"""This module displays a chart that show how many
apples, bananas, oranges and peaches that 3 poeple
Farrah, fred and Felicia have by colors and height of a
stack bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots a stacked bar graph of fruit quantities per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ["Farrah", "Fred", "Felicia"]
    fruits = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    bar_width = 0.5
    x = np.arange(len(people))
    bottom = np.zeros(len(people))

    for i in range(len(fruits)):
        plt.bar(
            x,
            fruit[i],
            width=bar_width,
            color=colors[i],
            label=fruits[i],
            bottom=bottom
        )
        bottom += fruit[i]

    plt.xticks(x, people)
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
