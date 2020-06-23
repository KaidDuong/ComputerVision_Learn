"""
Network diagram:    INPUT => CONV => RELU => FC
"""
#import the necessary libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        """
        Construct network

        Args:
            width (int): the width of the input images 
                        that will be used to train the network
            height (int): the height of the input images
            depth (int): the number of channels in the input image
            classes (int): the total number of classes that the network should learn to predict
        """
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape= (height, width, depth)

        #if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # build the shallowNet architecture
        # define the first CONV => RELU layer
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model