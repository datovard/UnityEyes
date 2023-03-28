
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau

class AlexNet:
    model = Model
    callbacks = []

    def __init__(self, width, height, depth, classes):
        #Instantiation
        self.model = Sequential()

        #1st Convolutional Layer
        self.model.add(Conv2D(filters = 96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #2nd Convolutional Layer
        self.model.add(Conv2D(filters = 256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #3rd Convolutional Layer
        self.model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        #4th Convolutional Layer
        self.model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        #5th Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #Passing it to a Fully Connected layer
        self.model.add(Flatten())
        # 1st Fully Connected Layer
        self.model.add(Dense(4096, input_shape=(32,32,3,)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        self.model.add(Dropout(0.4))

        #2nd Fully Connected Layer
        self.model.add(Dense(4096))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        #Add Dropout
        self.model.add(Dropout(0.4))

        #3rd Fully Connected Layer
        self.model.add(Dense(1000))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        #Add Dropout
        self.model.add(Dropout(0.4))

        #Output Layer
        self.model.add(Dense(9))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))
        self.callbacks = [
            ReduceLROnPlateau(   monitor='val_acc',   factor=.01,   patience=3,  min_lr=1e-5)
        ]

    def compileModel(self):
        self.model.compile(
            loss = keras.losses.categorical_crossentropy, 
            optimizer= 'adam', 
            metrics=['accuracy']
        )
    
    def fit(self, trainData, testData, augmentations, batchSize):
        train_generator = ImageDataGenerator(
            rotation_range = 2,
            zoom_range=.1)

        val_generator = ImageDataGenerator(
            rotation_range = 2, 
            zoom_range=.1)

        #Fitting the augmentation defined above to the data
        train_generator.fit(x_train)
        val_generator.fit(x_val)

        self.model.fit_generator(
            train_generator.flow(x_train, y_train, batch_size=batch_size), 
            epochs = epochs, 
            steps_per_epoch = x_train.shape[0]//batch_size, 
            validation_data = val_generator.flow(x_val, y_val, batch_size=batch_size), 
            validation_steps = 250, 
            callbacks = [lrr], 
            verbose=1
        )