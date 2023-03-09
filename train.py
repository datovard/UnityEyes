from keras.preprocessing.image import ImageDataGenerator
from src.models.GoogLeNet.MiniGoogleNet import MiniGoogleNet
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from src.utils.loadDataset import loadDataset

np.random.seed(1000)

IMAGE_SIZE = (48, 32, 1)
BATCH_SIZE = 64
NUM_EPOCHS = 100
INIT_LR = 5e-3

NUMBER_PROCESSES = 10
FILENAME = './input/results.csv'
INPUT = './input/train/Dataset 1/'
OUTPUT = './output/'

if __name__ == '__main__':
    X, y, y_labeled = loadDataset(INPUT, IMAGE_SIZE, NUMBER_PROCESSES)
    trainX, testX, trainY, testY = train_test_split(X, y_labeled, test_size=.3)

    augmentation = ImageDataGenerator(
        rotation_range = 2,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        fill_mode = "nearest")

    model = MiniGoogleNet(width = IMAGE_SIZE[1], height = IMAGE_SIZE[0], depth = IMAGE_SIZE[2], classes = 9)
    model.compileModel()
    history = model.fit((trainX, trainY), (testX, testY), augmentation, BATCH_SIZE)

    model.save(OUTPUT)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()
