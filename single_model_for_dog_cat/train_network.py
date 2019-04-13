# -*- coding:utf-8 _*-
"""
@author:liyige
@file:train_network.py
@time:2019/02/20
"""
import matplotlib
matplotlib.use("Agg")


from model.simplenet import Simplenet

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

from load_data import load_data
from predict import predict

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-e", "--epochs", type=int, default=10, help="# of epochs tp train network")
    ap.add_argument("-p", "--plot", type=str, default="plot.png")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    args = vars(ap.parse_args())
    return args

# initialize the number of epochs to train for, initial learning rate,
# and batch size
epochs = 20
lr = 1e-4
bs = 32
classes = 2


def train_network(trainX, testX, trainY, testY):
    # partition the data into training and testing splits using 60% of
    # the data for training and the remaining 40% for testing

    # initialize the model and optimizer
    print("[INFO] compiling model...")
    opt = Adam(lr=lr, decay=1e-4 / epochs)
    model = Simplenet.build(width=64, height=64, depth=1, classes=classes, reg=l2(0.0002))
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network for {} epochs...".format(epochs))
    H = model.fit(trainX, trainY, batch_size=bs, validation_data=(testX, testY),
                            epochs=epochs, verbose=1)
    # save the model to disk
    print("[INFO] serializing network...")
    model.save("cat_dog.model")


    # plot the training loss and accuracy
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")


if __name__ == '__main__':
    # args = args_parse()
    file_path = "dataset"
    (trainX, testX, trainY, testY) = load_data(file_path, classes)
    # train_network(trainX, testX, trainY, testY)
    predict(testX, testY, model_path='cat_dog.model')
