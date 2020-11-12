# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=2e-4):
        # initialize the model along with the input shape to be 
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape and
        # channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # BLOCK #1 :first CONV => RELU => BN => POOL => Dropout layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4),
            input_shape=inputShape, padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # BLOCK #2: second CONV => RELU => BN => POOL =>Dropout layer set
        model.add(Conv2D(256, (5, 5), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # BLOCK #3: [ CONV => RELU => BN ] * 3 => POOL => Dropout 
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => RELU => BN => Dropout layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        # BLOCK #4: second set of FC => RELU => BN => Dropout layers
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Desen(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        # return the reconstruct network architecture
        return model
