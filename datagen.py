# This module generates dataset of values for a two-variable function in CSV format

import numpy as np
import pandas as pd
import plot as plot


def func(x, y, a, b):
    #return (np.abs(x) / a) * np.cos(np.exp(y)) * np.sin(x ** 2 + y ** 2)
    return (np.sin(x) * np.cos(y))


def get_default_range():
    x = np.arange(-20, 20, step=0.5)  # define range of X
    y = np.arange(-20, 20, step=0.5)  # define range of Y
    return x, y


def generate(filename, show=False, save=True, add3d=False):
    x, y = get_default_range()
    a = 10
    b = 2
    xv, yv = np.meshgrid(x, y)  # create a mesh of all combinations of X and Y
    z = func(xv, yv, a, b)  # create ndarray of function values

    df = pd.DataFrame(
        {"x": xv.flatten(), "y": yv.flatten(), "z": z.flatten()})  # convert three ndarrays to pandas dataframe
    df.to_csv(filename)  # write to csv
    if show or save:
        plot.plot_dataset(filename, show=show, save=save, add3d=add3d)


# define default function to call when executed directly
if __name__ == '__main__':
    generate('dataset.csv')
