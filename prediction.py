# This module provides a visual way to test a model on a given set of coordinates
import pandas as pd
import plot


def predict(model, xy, epoch, show=False, save=False, add3d=False):
    if show or save:
        z = model.predict(xy[['x', 'y']])
        x = xy[['x']]
        y = xy[['y']]
        z = pd.DataFrame(z, columns=['z'])
        df = x.join(y)
        df = df.join(z)
        df = df.pivot('x', 'y', 'z')
        x = df.columns.values
        y = df.index.values
        z = df.values
        if show:
            plot.plot_points(x, y, z, add3d)
        if save:
            plot.plot_points_to_file(x, y, z, epoch, add3d=add3d)
