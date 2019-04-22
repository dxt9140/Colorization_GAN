import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Deconvolution2D


class Generator:

    @staticmethod
    def build():
        model = Sequential()
        # Downsampling portion
        # (32, 32)
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2), input_shape=(32, 32)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # (16, 16)
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # (8, 8)
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # (4, 4)
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # (2, 2)
        model.add(Conv2D(30, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Upsampling portion
        model.add(Deconvolution2D())
        model.add(Deconvolution2D())
        model.add(Deconvolution2D())
        model.add(Deconvolution2D())
        model.add(Deconvolution2D())


class Discriminator:

    @staticmethod
    def build():
        _ = None


# Ain't nobody got time for that
