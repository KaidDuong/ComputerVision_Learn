 
# import argparse
# from imutils import paths
# import cv2
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d","--dataset", required =True,
#     help="path to input dataset")
# args = vars(ap.parse_args())

# # grap the list of images tha we'll be describing
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))
# for (i,imagePath) in enumerate(imagePaths):
#             #load the image and extract the class label assuming
#             #that our path has the following format:
#             #/path/to/dataset/{class}/{image}.jpg
#             image = cv2.imread(imagePath)
#             fileName ="F:\datasets\Animals\pandas\panda.{}.jpeg".format(i)
#             print(fileName)
#             cv2.imwrite(fileName, image)

from sklearn.datasets import digits
data = digits.data
print(data.shape)