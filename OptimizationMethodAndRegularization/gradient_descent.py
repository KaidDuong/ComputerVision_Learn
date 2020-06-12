#import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import numpy as np
import argparse

# Define activation function
def sigmoid_activation(x):
    
    return 1.0 / (1+ np.exp(-x))

def predict(X, W):
    """
    Take the dot product between our fetures and weight matrix

    Args:
        X (matrix): the input data points
        W (matrix): the weights

    Returns:
        [matrix]: the predict values
    """
    
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs 
    # to binary class labels
    preds[preds <=0.5] = 0
    preds[preds > 0.5] = 1

    # return the predicts
    return preds

# contructs the argument parse and parse the arguments

ap =  argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args= vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points,
# where each data point is a 2D feature vector

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0],1)

# inset a column of 1's as the last entry in the feature matrix
# This little trick allow us to treat the bias 
# as a trainable parameter within the weight matrix
X= np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training with respect the remaining 50% for testing

(train_X, test_X, train_Y, test_Y) = train_test_split(X, y,
                                         test_size=0.5, random_state=42)


# intitialize our weight matrix and list of losses
print("[INFO] training ...")
W = np.random.randn(X.shape[1],1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]): 
    # take the dot product between our feature X and the
    # weight matrix W, then pass this value through our sigmoid 
    # activation function, thereby giving us our predicts on the dataset
    preds = sigmoid_activation(train_X.dot(W))

    # now that we have our predictions, we need to determine the error,
    # which is the difference between our predictions and the true value

    error = preds - train_Y
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot product between our fetures
    # and the error of the predictions
    gradient = train_X.T.dot(error)

    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the
    # term gradient descent by taking a small steptowards a set 
    # of more optimal parameters)
    W+= -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}"
                .format(int(epoch+1), loss))


# evaluate our model
print("[INFO] evaluating...")
preds = predict(test_X, W)
print(classification_report(test_Y, preds))

# plot the classification test data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(test_X[:, 0], test_X[:, 1], marker="o", c=test_Y, s=30)

# construct a  figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss") 
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.plot(np.arange(0, args["epochs"]), losses)
plt.show()

