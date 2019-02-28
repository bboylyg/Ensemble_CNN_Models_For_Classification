# -*- coding:utf-8 _*-
"""
@author:liyige
@file:SimpleNet.py
@time:2019/02/22
"""
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def SimpleNet(width, height, depth, classes):

    inputShape = (height, width, depth)
    model = Sequential()

    # first layer
    model.add(Conv2D(32, (11, 11), input_shape=inputShape, padding="same", kernel_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer
    model.add(Conv2D(64, (5, 5), padding="same", kernel_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third (final layer)
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # FC layer
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer="l2"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

    return model

