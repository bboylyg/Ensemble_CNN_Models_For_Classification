# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import argparse
import glob
import os
from scipy.stats import mode

def predict(testX, testY):

    # load the trained convolutional neural network
    print("[INFO] loading network...")

    modelPaths = os.path.sep.join(["save_models", "*.model"])
    modelPaths = list(glob.glob(modelPaths))
    models = []

    for (i, modelPath) in enumerate(modelPaths):
        print("[INFO] loading model {}/{}".format(i + 1, len(modelPaths)))
        print("Model path {}".format(i+1), modelPath)
        models.append(load_model(modelPath))

    # randomly select a few testing images and then initialize the output
    # set of images
    idxs = np.arange(0, testY.shape[0])
    idxs = np.random.choice(idxs, size=(25,), replace=False)
    images = []
    labelName = ['cat', 'dog']


    print("[INFO] evaluating ensemble...")
    predictions = []
    #遍历模型
    for model in models:
        # 模型预测
        predictions.append(model.predict(testX,batch_size=64))
        print(classification_report(testY.argmax(axis=1),predictions[0].argmax(axis=1), target_names=labelName))
    print("##############################################################")
    print("[INFO] Ensemble with Averaging")
    # 平均所有模型结果
    predictions = np.average(predictions,axis=0)
    # 模型结果
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),target_names=labelName))

    print("##############################################################")
    print('\n')
    print("[INFO] Ensemble with voting")
    # 投票法预测
    labels = []
    for m in models:
        predicts = np.argmax(m.predict(testX, batch_size=64), axis=1)
        labels.append(predicts)
    # print("labels_append:", labels)
    # Ensemble with voting
    labels = np.array(labels)
    # print("labels_array:", labels)
    # 矩阵转置，行变列，列变行（n*m→m*n）
    labels = np.transpose(labels, (1, 0))
    # print("labels_transpose:", labels)
    # scipy.stats.mode函数，返回传入数组/矩阵中最常出现的成员以及出现的次数
    # 两个模型预测值取众数（eg：[[0 0]]取[[0]]）
    labels = mode(labels, axis=1)[0]
    # print("labels_mode:", labels)
    labels = np.squeeze(labels)
    # print("labels: ", labels)
    print(classification_report(testY.argmax(axis=1),
                                labels, target_names=labelName))
