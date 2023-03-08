import pandas as pd
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
import numpy as np
import gc
# import multiprocessing

NUMBER_THREADS = 4

def divideChunks(list, chunkSize):
    for i in range(0, len(list), chunkSize):
        yield list[i:i + chunkSize]

def loadChunkFiles(dataset, chunk, imageSize):
    data = []
    for filename in chunk:
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
    return data

# def worker(procnum, send_end):
#     '''worker function'''
#     result = str(procnum) + ' - ' + str(os.getpid()) + ' represent!'
#     print(result)
#     send_end.send(result)

def datasetLoader(dataset, imageSize):
    data = []
    filenames = os.listdir(dataset)
    size = len(filenames)
    chunkSize = int(size/NUMBER_THREADS)

    chunks = list(divideChunks(filenames, chunkSize))
    if len(chunks) > NUMBER_THREADS:
        chunks.pop()

    # jobs = []
    # pipe_list = []
    # for i in range(NUMBER_THREADS):
    #     recv_end, send_end = multiprocessing.Pipe(False)
    #     process = multiprocessing.Process(target = worker, args = (i, send_end))
    #     jobs.append(process)
    #     pipe_list.append(recv_end)
    #     process.start()
    
    # for proc in jobs:
    #     proc.join()
    # result_list = [x.recv() for x in pipe_list]
    # print(result_list)
    
    print("LOADING DATASET - SIZE:", size)
    for chunk in chunks:
        data.extend(loadChunkFiles(dataset, chunk, imageSize))
        print("PERCENTAGE LOADED: %.2f%%" % ((len(data)/size) * 100), end='\r')  
    print("DATASET LOADED - IMAGES LOADED:", len(data))
    return data

def loadDataset(dataset, imageSize):
    df = pd.DataFrame(
        datasetLoader(dataset, imageSize), 
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

    return X, y, y_labeled