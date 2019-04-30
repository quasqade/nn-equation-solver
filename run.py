# This is an example that executes all steps inculding dataset generation, visualization, training and testing

import datagen
import prediction
import train
import argparse
import pandas as pd
import numpy as np


def run(existingweights=False, modelfile='', existingdataset=False, datasetfile='', epochs=500, earlystop=True,
        save=False, add3d=False):
    if existingweights:
        model = train.create_mlp(2)
        model.load_weights(modelfile)
        x, y = datagen.get_default_range()
        xv, yv = np.meshgrid(x, y)  # create a mesh of all combinations of X and Y
        df = pd.DataFrame({"x": xv.flatten(), "y": yv.flatten()})  # convert two ndarrays to pandas dataframe
        prediction.predict(model, df, 'evaluation', show=True, save=save, add3d=add3d)
    else:
        dataset = datasetfile
        if not existingdataset:
            dataset = 'dataset.csv'
            datagen.generate(dataset, show=True, save=save, add3d=add3d)
        train.train(dataset, epochs, earlystop, save=save, add3d=add3d)


# define default function to call when executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates and trains a new model or tests existing one.')
    parser.add_argument('--weights', nargs=1,
                        help='Specify a *.h5 file with weights to just test a model. If this argument is omitted, a new model will be created')
    parser.add_argument('--dataset', nargs=1,
                        help='Specify a *.csv dataset with 3 columns to skip generation. If this argument is omitted, a new dataset will be generated based on function in dataset.py')
    parser.add_argument('--epochs', type=int, nargs=1,
                        help='Specify how many epochs to train for')
    parser.add_argument('--earlystop', action='store_true',
                        help='Allows training to stop early when improvement slows down')
    parser.add_argument('--plot3d', action='store_true',
                        help='Adds a 3D plot to charts (may significantly slow down training)')
    parser.add_argument('--timelapse', action='store_true',
                        help='Will save timelapse of training process to IMG folder (may slow down training)')
    args = parser.parse_args()
    if args.weights:
        run(True, args.weights[0], save=args.timelapse,
            add3d=args.plot3d)  # TODO add ability to use axes from existing dataset
    else:
        if args.dataset:
            run(existingdataset=True, datasetfile=args.dataset, epochs=args.epochs[0], earlystop=args.earlystop,
                save=args.timelapse, add3d=args.plot3d)
        else:
            run(epochs=args.epochs[0], earlystop=args.earlystop, save=args.timelapse, add3d=args.plot3d)
