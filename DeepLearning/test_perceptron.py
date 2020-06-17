"""
    Applying the perceptron algorithm to bitwise OR dataset
"""
# import the necessary packages
from pyimagesearch.neural_network.perceptron import Perceptron
import numpy as np

# construct the OR dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])
def test_perceptron(X , y): 
    # define our perceptron and train it
    print("[INFO] training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)

    # now that our perceptron is trained we can evaluate it
    print("[INFO] testing perceptron...")

    # now tha our network is trained, loop over the data points
    for (x, target) in zip(X, y):
        # make a prediction on the data point and display the result to our console
        pred = p.predict(x)
        print("[INFO] data={}, grount-truth={}, pred={}".format
        (x, target[0], pred))

for (X, y) in zip((X_xor, X_and, X_xor), (y_or, y_and, y_xor)):
    print("-------------------------------\n")
    print(X)
    print(y)
    test_perceptron(X, y)