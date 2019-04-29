# This module generates dataset of values for a two-variable function in CSV format

import numpy as np
import pandas as pd


def func(x, y, a, b):
    return (np.abs(x)/a) * np.cos(np.exp(y)) * np.sin(x**2 + y ** 2)


def generate():
    x = np.arange(-3, 3, step=0.1)  # define range of X
    y = np.arange(-3, 3, step=0.1) # define range of Y
    a = 10
    b = 2
    xv, yv = np.meshgrid(x, y)  # create a mesh of all combinations of X and Y
    z = func(xv, yv, a, b)  # create ndarray of function values

    df = pd.DataFrame(
        {"x": xv.flatten(), "y": yv.flatten(), "z": z.flatten()})  # convert three ndarrays to pandas dataframe
    df.to_csv("dataset.csv")  # write to csv


# define default function to call when executed directly
if __name__ == '__main__':
    generate()
