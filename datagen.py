# This module generates dataset of values for a two-variable function in CSV format

import numpy as np
import pandas as pd


def func(x, y, a, b):
    return (np.sin(a * (b * (x ** 2) + b * (y ** 2)))) / a


def generate():
    x = np.arange(-5, 5, 0.1)  # define range of X
    y = x  # define range of Y
    a = 1
    b = 1
    xv, yv = np.meshgrid(x, y)  # create a mesh of all combinations of X and Y
    z = func(xv, yv, a, b)  # create ndarray of function values

    df = pd.DataFrame(
        {"x": xv.flatten(), "y": yv.flatten(), "z": z.flatten()})  # convert three ndarrays to pandas dataframe
    df.to_csv("dataset.csv", index=False)  # write to csv


# define default function to call when executed directly
if __name__ == '__main__':
    generate()
