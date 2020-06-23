# import the necessary libraries
from pyimagesearch.neural_network.cnn.minivggnet import MiniVGGNetwork
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.datasets import cifar10
import argparse
import os

import matplotlib
matplotlib.use("Agg")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help="path to the output directory")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}]".format(os.getpid()))

# load the training and testing set and then scale it to
# the range [0,1]
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()    
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)
y_train = lb.fit_transform(y_train)

# initialize the label names for the CIRFA10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model 
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNetwork.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of  callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath=figPath,jsonPath= jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(X_train, y_train, validation_data=(X_test, y_test),
    batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
