# This module splits dataset and trains a new network, saving weights
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import common
import plot
import outputobserver
import numpy as np


def plot_model(model):
    x, y, z = common.read('dataset.csv')


# Returns a multilayer perceptron model
def create_mlp(dimensions):
    model = Sequential()
    model.add(Dense(128, input_dim=dimensions, activation='relu'))  # input layer requires input_dim param
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation='linear'))  # sigmoid instead of relu for final probability
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
    return model


def split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)
    return train, test


def train(makemovie=False):
    df = pd.read_csv('dataset.csv')  # read indexed dataset
    train, test = split(df)  # split into test and training data
    model = create_mlp(2)  # create a model
    print(train.head())
    testcb = outputobserver.OutputObserver(df[['x', 'y']])
    history = model.fit(train[['x', 'y']], train[["z"]], epochs=1000, batch_size=20,
                        validation_data=(test[['x', 'y']], test[["z"]]), verbose=1, callbacks=[testcb])  # train
    plot.plot_history(history)


# define default function to call when executed directly
if __name__ == '__main__':
    train()
