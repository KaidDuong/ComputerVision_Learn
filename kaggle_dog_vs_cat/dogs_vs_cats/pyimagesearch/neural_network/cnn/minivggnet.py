# import the necessary libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNetwork:
    def build(width, height, depth, classes):
        """
        Initialize the network

        Args:
            width (int): the width of the input images
            height (init): the height of the input images
            depth (int): the number of channels of the input images
            classes (int): the number of the class labels in the classification task 
        """
        # initialize the model along with the input shape
        # to be channels last and the channel dimentions itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we're using the channels_firts ordering, update the input shape and channel dimentions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # define the first layer set : CONV => RELU => BN => CONV => RELU =>BN => POOL => DROPOUT
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # define the second layer set : (CONV => RELU=> BN) * 2 => POOL => DROPOUT
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same")) 
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # next define the firt (and only) set of FC => RELU=> BN => DROPOUT layer:
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # finally block softmax classifier: FC => SOFTMAX
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed netwrk architecture
        return model

