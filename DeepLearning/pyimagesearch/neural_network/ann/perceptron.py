"""  Perceptron algorithm process:
    1. Initalize our weight vector w with small random values
    2. Untill perceptron converges:
        2.1 Loop over each feature vector xi and true class label di in our training set D
        2.2 Take x and pass it through the network, calculating the ouput value by take the dot 
        product between input and weight and then followed pass through the step function
            yj = f(w(t)*xj)
        2.3 Update the weight vector by delta rule
            wi(t+1) = wi(t) + n(dj - yj)*xj,i for (0 <= i <= m)
"""
# import the necessary libraries
import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        """ Initialize Perceptron object

        Args:
            N (int): the number of columns in our input feature vectors.
            alpha (float, optional): our learning rate for the perceptron algorithm.
                                     Defaults to 0.1.
                                    Common choices of ones are normally in range [0.1, 0.01, 0.001]
        """
        # initialize the weight matrix and store the learning rate
        """ 
        Our weight matrix W with random values sampled from a “normal” (Gaussian)
        distribution with zero mean and unit variance. The weight matrix will have N +1 entries, one for
        each of the N inputs in the feature vector, plus one for the bias. We divide W by the square-root
        of the number of inputs, a common technique used to scale our weight matrix, leading to faster
        convergence
        """ 
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        """
        Apply the step function

        Args:
            x (float): the value of output of node
        
        Return: 1 if x > 0 and 0 otherwise
        """
        return 1 if x > 0 else 0
    
    def fit(self, X, y, epochs=10):
        """
        Fit a model to the data

        Args:
            X (matrix): the values of our actual training data
            y (matrix): the values of our target output class labes
            epochs (int, optional): the number of epochs our perceptron will train for
                                    . Defaults to 10.
        """
        # apply the bias trick by insert a column of ones into the training data,
        # which allows us to treat the bias as a trainable parameter directly inside the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs): 
            # loop over each individual data point
            for (x, target) in zip(X, y): 
                # take the dot product between the input features
                # and the weight matix, then pass this values through
                # the step function to obtain the prediction
                pred = self.step(np.dot(x, self.W))

                # only perform a weight update if our prediction
                # does not match the target: predict - target = 0
                if pred != target:
                    # determine the error
                    error = pred - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X , add_bias=True):
        """
        to predict the class lables for a given set of input data

        Args:
            X (matrix): the input feature data tha needs to be classified
            add_bias (bool, optional): . Verify that the bias column needs to be added

        Returns:
            matrix: the uotput predictions for input data 
        """
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if the bias column should be added
        if add_bias:
            # insert a column of 1's as the last entry in the feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between the input features and the weight
        # matrix , then pass the value through the step function
        return self.step(np.dot(X, self.W))