# -*- coding:utf-8 _*-
"""
@author:liyige
@file:LeNet5.py
@time:2019/02/22
"""
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, AveragePooling2D
from keras.optimizers import adam

def LeNet5(width, height, depth, classes):

    inputShape = (height, width, depth)
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=inputShape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

    return model