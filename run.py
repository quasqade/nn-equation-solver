# This is an example that executes all steps inculding dataset generation, visualization, training and testing

import datagen
import prediction
import train
import argparse
import pandas as pd
import numpy as np


def run(existingweights=False, filename='', save=False, add3d=False):
    if existingweights:
        model = train.create_mlp(2)
        model.load_weights(filename[0])
        x, y = datagen.get_default_range()
        xv, yv = np.meshgrid(x, y)  # create a mesh of all combinations of X and Y
        df = pd.DataFrame({"x": xv.flatten(), "y": yv.flatten()})  # convert two ndarrays to pandas dataframe
        prediction.predict(model, df, 'evaluation', show=True, save=save, add3d=add3d)
    else:
        datagen.generate('dataset.csv', show=True, save=save, add3d=add3d)
        train.train(save=save, add3d=add3d)


# define default function to call when executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates and trains a new model or tests existing one.')
    parser.add_argument('--weights', nargs=1,
                        help='Specify a *.h5 file with weights to just test a model. If this argument is omitted, a new model will be created')
    parser.add_argument('--plot3d', action='store_true',
                        help='Adds a 3D plot to charts (may significantly slow down training)')
    parser.add_argument('--timelapse', action='store_true',
                        help='Will save timelapse of training process to IMG folder (may slow down training)')
    args = parser.parse_args()
    if args.weights:
        run(True, args.weights, save=args.timelapse, add3d=args.plot3d)
    else:
        run(save=args.timelapse, add3d=args.plot3d)
