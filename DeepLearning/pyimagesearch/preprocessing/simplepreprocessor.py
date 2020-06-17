#import the necessary packages
import cv2

class SimplePreproccesor :
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        #store target image width, height and interpolation
        #method used when resizing
        self.width = width
        self.height = height 
        self.inter = inter
    
    def preprocess(self, image):
        #resize the image to a fixed size and ignoring the aspect ratio
        return cv2.resize(image, (self.height, self.width), 
                          interpolation =self.inter)
    
