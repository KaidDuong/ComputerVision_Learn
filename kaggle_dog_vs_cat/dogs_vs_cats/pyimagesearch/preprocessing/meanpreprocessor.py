# import the necessary libraries
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, Blue channel averages across a fixed size# training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the image into its respective Red, Green, Blue channels
        (B, G, R)= cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # mearge the normalized channels back together and return the resulting image
        return cv2.merge([B, G, R])

