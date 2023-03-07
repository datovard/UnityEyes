import pandas as pd
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
import numpy as np
import gc

def datasetLoader(dataset, size, imageSize):
    data = []
    print("LOADING DATASET - SIZE:", size)
    filenames = os.listdir(dataset)

    for filename in filenames:
        file = os.path.join(dataset, filename)
        image = Image.open(file)         
        number, quadrant = map(int, filename[:-4].split('_'))
        #imageCrop = image.crop(IMAGE_SQUARE_CROP)

        if(imageSize[2] == 1):
            imageCrop = image.convert('L')

        imageCrop.thumbnail(imageSize[:-1])

        data.append([
            number, 
            np.asarray(imageCrop, dtype="int32").flatten(), 
            quadrant - 1
        ])
        print("PERCENTAGE LOADED: %.2f%%" % ((len(data)/size) * 100), end='\r')  
    print("DATASET LOADED - IMAGES LOADED:", len(data))
    return data

def loadDataset(dataset, size, imageSize):
    df = pd.DataFrame(
        datasetLoader(dataset, size, imageSize), 
        columns=['number', 'image', 'quadrant']
    )
    print(df.head())

    df = df.sample(frac=1).reset_index(drop=True)
    # df = df.groupby('quadrant').apply(lambda x: x.sample(37))

    X = df.image.to_numpy()
    y = df.quadrant.to_numpy()

    del df
    gc.collect()

    X = np.asarray([ np.asarray(x.reshape(imageSize)) for x in X ])
    y = np.asarray([ np.asarray(cls.reshape(1)) for cls in y ])

    X = X/255
    y_labeled = to_categorical(y, 9)

    print(X.shape)
    print(y.shape)

    return X, y, y_labeled