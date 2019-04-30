# This module splits dataset and trains a new network, saving weights
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import plot
import outputobserver
import parameters as params


# Returns a multilayer perceptron model
def create_mlp(dimensions):
    model = Sequential()
    model.add(Dense(256, input_dim=dimensions, activation='relu'))  # input layer requires input_dim param
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(1, activation='linear'))  # relu instead of relu for final probability
    optimizer = keras.optimizers.Adam(lr=params.get_learning_rate())
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
    return model


def split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)
    return train, test


def train(dataset, epochs, earlystop, save=False, add3d=False):
    df = pd.read_csv(dataset)  # read indexed dataset
    train, test = split(df)  # split into test and training data
    model = create_mlp(2)  # create a model
    callbacks = []
    if save or add3d:
        # callback that predicts and saves images
        testcb = outputobserver.OutputObserver(df[['x', 'y']], save=save,
                                               add3d=add3d)
        callbacks.append(testcb)
    if earlystop:
        # callback that stops training when improvement stalls and chooses an epoch with best results
        earlystopcb = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                    patience=params.get_patience(),
                                                    restore_best_weights=True)
        callbacks.append(earlystopcb)

    history = model.fit(train[['x', 'y']], train[["z"]], epochs=epochs, batch_size=params.get_batch_size(),
                        validation_data=(test[['x', 'y']], test[["z"]]), verbose=1, callbacks=callbacks)  # train
    model.save('weights.h5')
    plot.plot_history(history, save=save)


# define default function to call when executed directly
if __name__ == '__main__':
    train()
