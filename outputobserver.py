import keras
import plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

class OutputObserver(keras.callbacks.Callback):

    def __init__(self, xy):
        self.xy = xy

    def on_epoch_end(self, epoch, logs={}):
        z = self.model.predict(self.xy[['x', 'y']])
        x = self.xy[['x']]
        y = self.xy[['y']]
        z = pd.DataFrame(z, columns=['z'])
        df = x.join(y)
        df = df.join(z)
        df = df.pivot('x', 'y', 'z')
        x = df.columns.values
        y = df.index.values
        z = df.values
        plot.plot_points_to_file(x,y,z,epoch)