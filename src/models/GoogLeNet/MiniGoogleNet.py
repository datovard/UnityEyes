from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, concatenate
from keras.models import Model
from keras.callbacks import LearningRateScheduler

NUM_EPOCHS = 100
INIT_LR = 5e-3

def polynomialDecay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

def conv_module(input,No_of_filters,filtersizeX,filtersizeY,stride,chanDim,padding="same"):
    input = Conv2D(No_of_filters,(filtersizeX,filtersizeY),strides=stride,padding=padding)(input)
    input = BatchNormalization(axis=chanDim)(input)
    input = Activation("relu")(input)
    return input

def inception_module(input,numK1x1,numK3x3,numk5x5,numPoolProj,chanDim):
    #Step 1
    conv_1x1 = conv_module(input, numK1x1, 1, 1,(1, 1), chanDim) 
    #Step 2
    conv_3x3 = conv_module(input, numK3x3, 3, 3,(1, 1), chanDim)
    conv_5x5 = conv_module(input, numk5x5, 5, 5,(1, 1), chanDim)
    #Step 3
    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    pool_proj = Conv2D(numPoolProj, (1, 1), padding='same', activation='relu')(pool_proj)
    #Step 4
    input = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=chanDim)
    return input

def downsample_module(input, No_of_filters, chanDim):
    conv_3x3 = conv_module(
        input,
        No_of_filters,
        3,
        3,
        (2, 2),
        chanDim,
        padding = "valid"
    )
    pool = MaxPooling2D((3,3),strides=(2,2))(input)
    input = concatenate([conv_3x3,pool], axis=chanDim)
    return input

def MiniGoogleNet(width, height, depth, classes):
    inputShape = (height, width, depth)
    chanDim = -1

    # (Step 1) Define the model input
    inputs = Input(shape = inputShape)

    # First CONV module
    x = conv_module(inputs, 96, 3, 3, (1, 1),chanDim)

    # (Step 2) Two Inception modules followed by a downsample module
    x = inception_module(x, 32, 32,32,32,chanDim)
    x = inception_module(x, 32, 48, 48,32,chanDim)
    x = downsample_module(x, 80, chanDim)

    # (Step 3) Five Inception modules followed by a downsample module
    x = inception_module(x, 112, 48, 32, 48,chanDim)
    x = inception_module(x, 96, 64, 32,32,chanDim)
    x = inception_module(x, 80, 80, 32,32,chanDim)
    x = inception_module(x, 48, 96, 32,32,chanDim)
    x = inception_module(x, 112, 48, 32, 48,chanDim)
    x = downsample_module(x, 96, chanDim)

    # (Step 4) Two Inception modules followed
    x = inception_module(x, 176, 160, 96, 96, chanDim)
    x = inception_module(x, 176, 160, 96, 96, chanDim)

    # Global POOL and dropout
    x = AveragePooling2D((7, 7))(x)
    x = Dropout(0.5)(x)

    # (Step 5) Softmax classifier
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    # Create the model
    model = Model(inputs, x, name = "googlenet")
    return model

def miniGoogleNetCompiled(width, height, depth, classes):
    callbacks=[LearningRateScheduler(polynomialDecay)]

    optimizer = SGD(lr = INIT_LR, momentum = 0.9)

    model = MiniGoogleNet.MiniGoogleNet(width = IMAGE_SIZE[1], height = IMAGE_SIZE[0], depth = IMAGE_SIZE[2], classes = 9)
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = ["accuracy"])

    return model