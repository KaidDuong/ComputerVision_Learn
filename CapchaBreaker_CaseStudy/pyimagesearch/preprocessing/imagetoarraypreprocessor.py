# import the necessary libraries
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, data_format = None):
        # store the image data format
        self.data_format = data_format

    def preprocess(self, image):
        """
        Apply the Keras utility fuction that correctly rearranges
        the dimensions of the image

        Args:
            image : the input image
        """
        return img_to_array(image, data_format=self.data_format)