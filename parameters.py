# Common parameters for easy tweaking
import numpy as np


def get_axes_range():
    x = np.arange(-3, 3, step=0.1)  # define range of X
    y = np.arange(-3, 3, step=0.1)  # define range of Y
    return x, y


def get_learning_rate():
    return 0.0001


def get_patience():
    return 20


def get_batch_size():
    return 10
