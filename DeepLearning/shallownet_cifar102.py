# import the necessary libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.neural_network.cnn.shallownet import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np 
import matplotlib.pyplot as plt

# load the training and testing data, then scale it into the
# range[0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the lable names for the cifar-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=32, epochs=40, verbose=1)

# evaluate the Network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40),H.history['loss'], labels="loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], labels="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], labels="train_accuracy")
plt.plot(np.arange(0, 40),H.history["val_accuracy"], labels ="val_accuracy")
plt.title("The Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")