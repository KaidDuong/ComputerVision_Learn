from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreproccesor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.neural_network.cnn.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import argparse
import matplotlib.pyplot as plt

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m","--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())

#grap the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# intitialize th image preprocessors
sp = SimplePreproccesor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw intensities
# to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
print("shape data: ", data.shape)
print("data: ",data[0])
print("label: ", labels[0])

# partition th data in to the training and testing splits using 75% of the data
#  for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the lables from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the label names
labelNames = ["cat", "dog","panda"]

# initialize the optimizer and model
print ("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=len(labelNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the Network
print ("[INFO] training Network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# save the network to disk
print ("[INFO] Serializing network...")
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

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
plt.show()