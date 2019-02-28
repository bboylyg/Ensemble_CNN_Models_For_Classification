#encoding:utf-8
# 设置图像的背景
import matplotlib
matplotlib.use('agg')

# 加载模块
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from preprocessing.load_data import load_data
from models.SimpleNet import SimpleNet
from models.LeNet5 import LeNet5
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.optimizers import Adam
from test_ensemble import predict
import os

# initialize the number of epochs to train for, initial learning rate,
# and batch size
epochs = 30
lr = 1e-4
bs = 32
reg = 0.002
width, height, depth, classes = 64, 64, 1, 2
# 初始化标签名陈
labelNames = ["cat", "dog"]

# 初始化数据增强模块
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode='nearest')

def train_models(trainX, testX, trainY, testY):

    print("[INFO] initialize models")
    models = []

    models.append(SimpleNet(width, height, depth, classes))
    models.append(LeNet5(width, height, depth, classes))
    print("[INFO] the number of models :", len(models))

    Hs = []
    # 遍历模型训练个数
    for i in np.arange(0, len(models)):
        # 初始化优化器和模型
        print("[INFO] training model {}/{}".format(i+1, len(models)))
        # opt = Adam(lr=lr, decay=1e-4 / epochs)
        # opt = SGD(lr=0.001,decay=0.01/ 40,momentum=0.9,
        #           nesterov=True)

        # 训练网络
        H = models[i].fit_generator(aug.flow(trainX, trainY, batch_size=bs),
                                validation_data=(testX, testY), epochs=epochs,
                                steps_per_epoch=len(trainX) // bs, verbose=1)
        # 将模型保存到磁盘中
        p = ['save_models',"model_{}.model".format(i)]
        models[i].save(os.path.sep.join(p))

        Hs.append(models[i])

        # plot the training loss and accuracy
        N = epochs
        p = ['model_{}.png'.format(i)]
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, N), H.history['loss'],
                 label='train_loss')
        plt.plot(np.arange(0, N), H.history['val_loss'],
                 label='val_loss')
        plt.plot(np.arange(0, N), H.history['acc'],
                 label='train-acc')
        plt.plot(np.arange(0, N), H.history['val_acc'],
                 label='val-acc')
        plt.title("Training Loss and Accuracy for model {}".format(i))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(os.path.sep.join(p))
        plt.close()


if __name__ == '__main__':
    # args = args_parse()
    file_path = "dataset"
    (trainX, testX, trainY, testY) = load_data(file_path, classes)
    # train_models(trainX, testX, trainY, testY)
    predict(testX, testY)

