
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from src.utils.loadDataset import loadDataset

IMAGE_SIZE = (32, 32, 1)

NUMBER_PROCESSES = 10
DATASET = './input/test/Real Test 2/'
MODEL = './output/models/miniGoogLeNet-20230324-193709.h5'

def getPredictedClass(predictions):
    y_pred_class = []
    print(predictions)
    for prediction in predictions:
        result = np.where(prediction == np.amax(prediction))[0]
        y_pred_class.append(result[0])
    
    return y_pred_class

def testModel(modelName, dataset, imageSize, numberProcesses, classSample = None):
    X_test, y_test, y_labeled = loadDataset(dataset, imageSize, numberProcesses, classSample)

    model = load_model(modelName)
    score = model.evaluate(X_test, y_labeled)

    y_pred = getPredictedClass(model.predict(X_test))

    confusion_mtx = confusion_matrix(y_test, y_pred)
    return [score, confusion_mtx]

if __name__ == '__main__':
    response = testModel(MODEL, DATASET, IMAGE_SIZE, NUMBER_PROCESSES)

    print('Test Score = ', response[0][0])
    print('Test Accuracy = ', response[0][1])
    print(response[1])
