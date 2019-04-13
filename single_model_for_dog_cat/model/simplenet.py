# -*- coding:utf-8 _*-
"""
@author:liyige
@file:simplenet.py
@time:2019/02/20
"""
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense)
from keras.regularizers import l2
from keras import backend as K

class Simplenet:
    @staticmethod
    def build(width, height, depth, classes, reg):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # first layer
        model.add(Conv2D(64, (11, 11), input_shape=inputShape,
                         padding="same", kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # second layer
        model.add(Conv2D(128, (5, 5), padding="same",
                         kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # third (final layer)
        model.add(Conv2D(256, (3, 3), padding="same",
                         kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # FC layer
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

