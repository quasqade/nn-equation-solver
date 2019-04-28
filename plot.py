# This module plots a generated 3 column dataset

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np


def read(filename):
    df = pd.read_csv(filename)
    df = df.pivot('x', 'y', 'z')
    x = df.columns.values
    y = df.index.values
    z = df.values
    return x, y, z


def plot():
    x, y, z = read('dataset.csv')
    xv, yv = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(xv, yv, z)
    ax.set_xlabel('Surface map')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xv, yv, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('3D plot')
    plt.show()


if __name__ == '__main__':
    plot()
