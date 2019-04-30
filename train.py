# This module splits dataset and trains a new network, saving weights
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import plot
import outputobserver


# Returns a multilayer perceptron model
def create_mlp(dimensions):
    model = Sequential()
    model.add(Dense(256, input_dim=dimensions, activation='relu'))  # input layer requires input_dim param
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(1, activation='linear'))  # relu instead of relu for final probability
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
    return model


def split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)
    return train, test


def train(dataset, save=False, add3d=False):
    df = pd.read_csv('dataset.csv')  # read indexed dataset
    train, test = split(df)  # split into test and training data
    model = create_mlp(2)  # create a model
    testcb = outputobserver.OutputObserver(df[['x', 'y']], save=save, add3d=add3d)
    history = model.fit(train[['x', 'y']], train[["z"]], epochs=500, batch_size=10,
                        validation_data=(test[['x', 'y']], test[["z"]]), verbose=1, callbacks=[testcb])  # train
    model.save('weights.h5')
    plot.plot_history(history,save=save)


# define default function to call when executed directly
if __name__ == '__main__':
    train()
