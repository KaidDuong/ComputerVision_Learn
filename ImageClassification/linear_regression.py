# Import the necessary libraries
import numpy as np 
import cv2
import argparse

# contruct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
                help= "path to the image")
args =vars(ap.parse_args())

# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results
labels = ["dog", "cat"]
np.random.seed(1)

# randomly initialize our weigth matrx and bias vestor 
# in a real training and classificatio task, these parameters would
# be learn by our model
W = np.random.randn(3,3072)
b=np.random.randn(3)


# load our input image , resize it, and then flattan it into
# our feature vector representaton

orig = cv2.imread(args["image"])
image = cv2.resize(orig,(32,32)).flatten()

# computr the output scores by taking the dot poduct between 
# the weight matrix and image pixels, followed by adding the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))

# draw the label with the highest score on the mage as our prediction
cv2.putText(orig,"Label: {}".format(labels[np.argmax(scores)]),
(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255 ,0),2)

# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)