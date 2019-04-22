import keras
import numpy as np
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU
from keras.models import Sequential


class CNN:

    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(30, kernel_size=(4, 4), input_shape=(32, 32, 1), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(1, 1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(3072, activation='tanh'))

        return model


if __name__ == '__main__':
    _ = None
