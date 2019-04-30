import keras
import prediction


class OutputObserver(keras.callbacks.Callback):
    save = False
    add3d = False

    def __init__(self, xy, save=False, add3d=False):
        self.xy = xy
        self.save = save
        self.add3d = add3d

    def on_epoch_end(self, epoch, logs={}):
        prediction.predict(self.model, self.xy, epoch, save=self.save, add3d=self.add3d)
