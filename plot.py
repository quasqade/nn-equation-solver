# This module provides visualization tools

import common
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def plot_points(x, y, z):
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

def plot_points_to_file(x, y, z, epoch):
    xv, yv = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(xv, yv, z)
    ax.set_xlabel('Surface map')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xv, yv, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('3D plot')
    #img_dir = os.makedirs(".\\IMG")
    plt.savefig('.\\IMG\\'+str(epoch)+'.png')


def plot_dataset():
    x, y, z = common.read('dataset.csv')
    plot_points(x,y,z)



def plot_history(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# define default function to call when executed directly
if __name__ == '__main__':
    plot_dataset()
