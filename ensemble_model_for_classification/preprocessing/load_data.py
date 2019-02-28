# -*- coding:utf-8 _*-
"""
@author:liyige
@file:load_data.py
@time:2019/02/20
"""
from imutils import paths
import cv2
import os
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def load_data(path, classes):
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in tqdm(imagePaths):
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))

        # update the data and labels lists respectively
        data.append(image)
        labels.append(label)

    # convert the data into a Numpy array
    # Scaling all pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    # reshape the data dimension(add a channel dimension)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

    # encode the labels(string) as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, classes)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.40, stratify=labels,
                                                      random_state=42)
    return (trainX, testX, trainY, testY)