# Common functions that are used on multiple steps

import pandas as pd


# Reads dataset from filename and returns tuple of 3 ndarrays
def read(filename):
    df = pd.read_csv(filename)
    df = df.pivot('x', 'y', 'z')
    x = df.columns.values
    y = df.index.values
    z = df.values
    return x, y, z
