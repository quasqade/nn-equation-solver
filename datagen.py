# This module generates dataset of values for a two-variable function in CSV format

import numpy as np
import pandas as pd
import plot
import parameters


def func(x, y, a, b):
    #return (np.abs(x) / a) * np.cos(np.exp(y)) * np.sin(x ** 2 + y ** 2)
    return np.sin(x**2+y**2)


def generate(filename, distribution='uniform', N=parameters.get_points(), show=False, save=True, add3d=False):
    min, max = parameters.get_axes_range()
    if distribution == 'uniform':
        x = np.linspace(min, max, N)
        y = np.linspace(min, max, N)
    elif distribution == 'random':
        x = np.random.uniform(min,max,N)
        y = np.random.uniform(min,max,N)

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
