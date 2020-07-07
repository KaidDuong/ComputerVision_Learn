# import the necessary libraries
import cv2

class ConverBGRToGrayScalePreprocessor:
    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)