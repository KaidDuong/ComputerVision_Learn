"""
    1. Load the CIRFA10 dataset from disk
    3. Initialize the MiniVGGNetwork architecture
    3. Training MiniVGGNetwork using the training set
    4. Evaluating MiniVGGNetwork performance with the testing set
"""
# import the necessary libraries

# set the matplot backend so that figures can be saved on the background
import matplotlib
matplotlib.use("Agg")

from pyimagesearch.neural_network.cnn.minivggnet import MiniVGGNetwork
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help="path to the loss/accuracy figure")
args = vars(ap.parse_args())

# grap the CIRFA10 dataset from disk and scale data to the
# range [0, 1]
((X_train,y_train),(X_test, y_test)) = cifar10.load_data()
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# convet the labels from integers to vectors
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)
y_train = lb.fit_transform(y_train)

# initialize label names for CIRFA10 dataset
labelNames =["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]
labelNames2 = [str(x) for x in lb.classes_]
print(labelNames2)

# initialize the optimizers and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNetwork.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64,
    epochs=40, verbose=1)
# evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(X_test, batch_size=64).argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), preds, target_names=labelNames))

# plot the training loss and Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40),H.history['loss'], label="loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 40),H.history["val_accuracy"], label ="val_accuracy")
plt.title("The Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# save the fig
plt.savefig(args["output"])
plt.close()

