
from src.models.GoogLeNet.MiniGoogleNet import MiniGoogleNet
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils.loadDataset import loadDataset
from src.utils.writeReport import writeRecordOnReport
from test import testModel

np.random.seed(1000)

DEFAULT_VALUES = {
    "imageSize": (32, 32, 1),
    "batchSize": 64,
    "epochSize": 100,
    "learningRate": 5e-3,
    "momentum": 0.9,
    "augmentation": {
        "rotation_range": 2,
        "width_shift_range": 0.3,
        "height_shift_range": 0.3,
        "fill_mode": "nearest"
    },
    "tests": []
}

DEFAULT_TEST = [
    {
        "dataset": './input/test/Real Test/',
        "classSample": 30
    },
    {
        "dataset": './input/test/Real Test 2/',
        "classSample": None
    }
]

NUMBER_PROCESSES = 10
FILENAME = './input/results.csv'
CLASS_SAMPLE = 50
INPUTS = {}

def loadInputs():
    global INPUTS
    cases = []

    for i in range(10):
        copied = DEFAULT_VALUES.copy()

        j = -3 * np.random.random()
        learningRate = 10 ** j
        copied["learningRate"] = learningRate
        copied["tests"] = DEFAULT_TEST

        cases.append(copied)
    
    INPUTS["./input/train/Dataset 1/"] = cases

OUTPUT = './output/models/'
TEST_INPUT = '' # './input/test/Real Test/'
LOGS_OUTPUT = "./output/logs/fit/"

if __name__ == '__main__':
    loadInputs()
    for dataset in INPUTS:
        for configuration in INPUTS[dataset]:
            print("LOADING DATASET:", dataset)
            X, y, y_labeled = loadDataset(dataset, configuration["imageSize"], NUMBER_PROCESSES, CLASS_SAMPLE)

            if "hardTest" in configuration:
                print("USING HARD TEST")
                trainX, trainY = X, y_labeled
                testX, _, testY = loadDataset(configuration["hardTest"], configuration["imageSize"], NUMBER_PROCESSES)
            else:
                trainX, testX, trainY, testY = train_test_split(X, y_labeled, test_size=.3)

            model = MiniGoogleNet(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                momentum = configuration["momentum"],
                augmentations = configuration["augmentation"],
                logsOutput = LOGS_OUTPUT
            )

            model.compileModel()
            
            history = model.fit((trainX, trainY), (testX, testY))
            
            model.save(OUTPUT)

            testsResults = []
            for test in configuration["tests"]:
                response = testModel(
                    './output/models/' + model.getName() + '.h5',
                    test["dataset"],
                    configuration["imageSize"],
                    NUMBER_PROCESSES,
                    test["classSample"]
                )

                testsResults.append({
                    "dataset": test["dataset"],
                    "scores": { 
                        "score": response[0][0],
                        "accuracy": response[0][1],
                    },
                    "matrix": response[1].tolist()
                })

            writeRecordOnReport(
                model.getName(),
                dataset,
                configuration["imageSize"],
                configuration["epochSize"],
                configuration["learningRate"],
                configuration["batchSize"],
                configuration["augmentation"],
                model.getSpecialValues(configuration),
                testsResults)
