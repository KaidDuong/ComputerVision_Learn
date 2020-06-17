
# import the nessary libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from pyimagesearch.neural_network.nn_backpropagation_SGD import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

# load the MNIST dataset and aply min/max scaling to scale the
# the pixel intensity values to the range [0, 1]( each image is
#  represented by an 8 x 8 = 64-dim feature vector)

print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
    data.shape [1]))

# construct the trainning and testing dataset
(train_X, test_X, train_Y, test_Y)= train_test_split(data, 
    digits.target, test_size=0.25)

# convert the labels from intergers to vectors
train_Y = LabelBinarizer().fit_transform(train_Y)
test_Y = LabelBinarizer().fit_transform(test_Y)

# train the network
print("[INFO] training network...")
# initialize the network with 64 -32 -16 -10 architecture
nn = NeuralNetwork([train_X.shape[1], 32, 16, 10])
nn.fit(train_X, train_Y, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(test_X)
predictions = predictions.argmax(axis=1)
print(classification_report(test_Y.argmax(axis=1), predictions))

# plot the training loss 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 1000), nn.losses,label="train_loss")
plt.title("Training loss ")
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show