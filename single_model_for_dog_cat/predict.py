# -*- coding:utf-8 _*-
"""
@author:liyige
@file:predict.py
@time:2019/02/20
"""
# import the necessary packages
from keras.models import load_model
from imutils import build_montages
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2


def predict(testX, testY, model_path):

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model_path)
    # randomly select a few testing images and then initialize the output
    # set of images
    idxs = np.arange(0, testY.shape[0])
    idxs = np.random.choice(idxs, size=(25,), replace=False)
    images = []
    labelName = ['cat', 'dog']
    # loop over the testing indexes
    for i in idxs:
        # grab the current testing image and classify it
        image = np.expand_dims(testX[i], axis=0)
        preds = model.predict(image)
        j = preds.argmax(axis=1)[0]

        label = labelName[j]

        # rescale the image into the range [0, 255] and then resize it so
        # we can more easily visualize it
        output = (image[0] * 255).astype("uint8")
        output = np.dstack([output] * 3)

        output = cv2.resize(output, (128, 128))

        # draw the colored class label on the output image and add it to
        # the set of output images
        label_color = (0, 0, 255) if "non" in label else (0, 255, 0)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_color, 2)
        images.append(output)

    # create a montage using 128x128 "tiles" with 5 rows and 5 columns
    montage = build_montages(images, (128, 128), (5, 5))[0]

    # show the output montage
    cv2.imshow("Output", montage)
    cv2.waitKey(0)