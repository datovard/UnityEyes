
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from src.utils.loadDataset import loadDataset

IMAGE_SIZE = (48, 32, 1)

NUMBER_PROCESSES = 1
DATASET = './input/test/'
MODEL = './output/modelo_GoogLeNet_FULL.h5'

if __name__ == '__main__':
    X_test, y_test, y_labeled = loadDataset(DATASET, IMAGE_SIZE, NUMBER_PROCESSES)

    model = load_model(MODEL)

    score = model.evaluate(X_test, y_labeled)
    print('Test Score=', score[0])
    print('Test Accuracy=', score[1])

    y_pred = model.predict(X_test)
    y_true = y_test

    y_pred_class = []
    for pred in y_pred:
        max_value = np.amax(pred)
        result = np.where(pred == np.amax(pred))[0]
        y_pred_class.append(result[0])

    print(y_pred_class[0])
    print(y_true[0])

    confusion_mtx = confusion_matrix(y_true, y_pred_class)
    print(confusion_mtx)
    # class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    # Plotting non-normalized confusion matrix
    # plot_confusion_matrix(y_true, y_pred_class, classes = class_names,title = 'Confusion matrix, without normalization')