# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        """
        Building the network

        Args:
            width (int): the width of the input images
            height (int): the height of the input images
            depth (int): the number of channels of the input images
            classes (int): the number of class labels in the classification task
        """
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using the "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # Begining to build the architecture of Network
        # define the first set CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set CONV => RELU => POOL
        model.add(Conv2D(50, (14, 14), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


    