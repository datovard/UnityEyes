from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import os
from src.models.GoogLeNet import MiniGoogleNet
from src.utils.loadDataset import loadDataset

np.random.seed(1000)

IMAGE_SIZE = (48, 32, 1)
BATCH_SIZE = 64
NUM_EPOCHS = 100
INIT_LR = 5e-3

FILENAME = './input/results.csv'
DATASET = './input/train/Dataset 1/'

def polynomialDecay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

if __name__ == '__main__':
    X, y, y_labeled = loadDataset(DATASET, IMAGE_SIZE)
    # trainX, testX, trainY, testY = train_test_split(X, y_labeled, test_size=.3)

    # augmentation = ImageDataGenerator(
    #     rotation_range = 2,
    #     width_shift_range = 0.3,
    #     height_shift_range = 0.3,
    #     fill_mode = "nearest")

    # callbacks = [LearningRateScheduler(polynomialDecay)]

    # optimizer = SGD(lr = INIT_LR, momentum = 0.9)

    # model = MiniGoogleNet.MiniGoogleNet(width = IMAGE_SIZE[1], height = IMAGE_SIZE[0], depth = IMAGE_SIZE[2], classes = 9)
    # model.compile(
    #     loss = "categorical_crossentropy",
    #     optimizer = optimizer,
    #     metrics = ["accuracy"])

    # history = model.fit(
    #     augmentation.flow(trainX, trainY, batch_size = BATCH_SIZE),
    #     validation_data = (testX, testY),
    #     steps_per_epoch = len(trainX) // BATCH_SIZE,
    #     epochs = NUM_EPOCHS, 
    #     callbacks = callbacks, 
    #     verbose = 1)

    # model.save('./output/modelo_GoogLeNet_FULL.h5')

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc = 'upper left')
    # plt.show()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc = 'upper left')
    # plt.show()
