#!/usr/bin/env python3
"""This module contains a function that draws a plot
as a histogram"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """This function shows a frequency a histogram
    chart"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(range(0, 101, 10))
    plt.hist(student_grades, bins=range(0, 110, 10), edgecolor="black")

    plt.show()
