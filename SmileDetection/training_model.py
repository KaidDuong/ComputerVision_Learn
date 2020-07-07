#import the necessary libraries
from pyimagesearch.neural_network.cnn.minivggnet import MiniVGGNetwork
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from imutils import paths
import  argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import os

# construct the argument parse and parse the arguments : 
# --dataset : the path to the SMILES dataset residing on disk
# --model : the path to where the serialized Network weights will be saved after training
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True,
    help="path to the input dataset of faces")
ap.add_argument("-m","--model", required=True,
    help="path to the trained network")
ap.add_argument("-f","--figure", required=True,
    help="path to the loss and accuracy plot ")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []
imagePaths = list(paths.list_images(args["dataset"]))
# loop over the input images
for (i,imagePath) in enumerate(sorted(imagePaths)):
# load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)
    #show an update every verbose images
    if i >0 and (i+1)% 100 == 0:
        print("[INFO] process {}/{}".format(i+1, len(imagePaths)))

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.20, stratify=labels, random_state=42)

#initialize the model
print("[INFO] compiling model...")
model = MiniVGGNetwork.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
    metrics=["accuracy"])

# construct the set of  callbacks
# initialize the training monitor
figPath = os.path.sep.join([args["figure"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["figure"], "{}.json".format(os.getpid())])
training_monitor = TrainingMonitor(figPath=figPath,jsonPath= jsonPath)

# initialize the checkpoint improvements
checkpoint_improvements = ModelCheckpoint(args["model"], monitor="val_loss",
    mode="min", save_best_only=True, verbose=1)

# initialize the callbacks
callbacks = [training_monitor, checkpoint_improvements]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    class_weight=classWeight, callbacks=callbacks, batch_size=64, epochs=15, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=le.classes_))

