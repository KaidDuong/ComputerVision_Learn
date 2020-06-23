"""
    1. loading the mnist dataset from disk
    2. iniitiating the LeNet architecture
    3. Training LeNet
    4.Evaluating network performence
"""

# import the necessary libraries
from pyimagesearch.neural_network.cnn.lenet import LeNet
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help="path to output figure")
args = vars(ap.parse_args())

# grap the MNIST dataset 
print("[INFO] accessing MNIST...")

dataset = datasets.fetch_openml('mnist_784', version=1)
data = dataset.data

# if we are using the "channels first" ordering, then reshape the 
# design matrix such that the matrix is
# num_examples x depth x rows x columns

if K.image_data_format == "channels_first":
    data = np.reshape(data,(data.shape[0], 1, 28, 28))

# otherwise, we are using the "channels_last" ordering, so the design matrix shape should be: 
# num_examples x rows x columns x depth
else:
    data = np.reshape(data,(data.shape[0], 28, 28, 1))

print(data.shape)
# scale th input data to the range[0,1]
data = data / 255.0
labels = dataset.target.astype("int")

# partition the data into training and testing splits using
# 75% data for training and the remaining 25% for testing
(X_train, X_test ,y_train, y_test) = train_test_split(data, labels,
      test_size=0.25, random_state=42)

# convet the labels from integers to vectors
lb= LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
labelNames = [ str(x) for x in lb.classes_]

# initialize the optimizer and model
print("[INFO] compiling network...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",
    optimizer=opt, metrics=["accuracy"])

# train the Network
print("[INFO] training Network...")
H = model.fit(X_train, y_train, validation_data= (X_test, y_test),
    epochs=40, batch_size=128, verbose=1)
# evaluate the Network
print("[INFO] evaluating the network...")
preds = model.predict(X_test, batch_size=128).argmax(axis=1)
print(classification_report(y_test.argmax(axis=1),
    preds, 
    target_names=labelNames))

# plot the training loss and accuracy
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