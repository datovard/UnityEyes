
import os

REPORT_OUTPUT = './output/reports/report.csv'

def writeRecordOnReport(name, dataset, shape, epochs, learningRate, batchSize, augmentations, specialValues, tests):
    file = open(REPORT_OUTPUT, 'a+')

    if(os.stat(REPORT_OUTPUT).st_size == 0):
       file.write("name;dataset;shape;epochs;learningRate;batchSize;augmentation;specialValues;tests\n")

    record = ";".join([
        name,
        dataset,
        "(" + ",".join([str(value) for value in shape]) + ")",
        str(epochs),
        str(learningRate), 
        str(batchSize),
        str(augmentations),
        str(specialValues),
        str(tests).strip()
    ])
    
    file.write(record + "\n")