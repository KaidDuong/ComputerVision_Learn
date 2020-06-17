
# import the necessary libraries
from pyimagesearch.neural_network.nn_backpropagation import NeuralNetwork
import numpy as np

# construct the XOR datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y= np.array([[0],[1],[1],[0]])

# define our 2-2-1 neural network and train it
nn = NeuralNetwork([2,2,1], alpha=0.5)
nn.fit(X ,y, epochs=10000)

# loop over the XOR data points
for (x, target) in zip(X, y):
    # make a predict on the data point and display 
    # the result on our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, grount-truth={}, pred={:.4f}, step={}".format(
        x, target[0], pred, step
    ))