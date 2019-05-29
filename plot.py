# This module provides visualization tools

import parameters
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd


def get_img_dir():
    img_dir = ".\\IMG"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    return img_dir


def get_plot(x, y, z, add3d=False):
    xv, yv = np.meshgrid(x, y)

    if add3d:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure()
        ax = plt.gca()

    ax.contourf(xv, yv, z)
    ax.set_xlabel('Surface map')

    if add3d:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(xv, yv, z, rstride=1, cstride=1, cmap=cm.viridis,
                        linewidth=0, antialiased=False)
        ax.set_xlabel('3D plot')

    return plt


def plot_points(x, y, z, add3d=False):
    plt = get_plot(x, y, z, add3d)
    plt.show()


def plot_points_to_file(x, y, z, epoch, add3d=False):
    plt = get_plot(x, y, z, add3d)
    plt.savefig(get_img_dir() + '\\' + str(epoch) + '.png')
    plt.close('all')


def plot_dataset(filename, add3d=False, save=True, show=False):
    df = pd.read_csv(filename)
    df = df.pivot('x', 'y', 'z')
    x = df.columns.values
    y = df.index.values
    z = df.values
    if (save):
        plot_points_to_file(x, y, z, epoch='source', add3d=add3d)
    if (show):
        plot_points(x, y, z, add3d)


def plot_history(history, save=True):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if save:
        plt.savefig(get_img_dir() + '\\' + 'history.png')
    plt.show()


# define default function to call when executed directly
if __name__ == '__main__':
    plot_dataset()
