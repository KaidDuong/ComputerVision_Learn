"""
    Implementing backpropagation algorithm
"""
# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        """
        Constructor

        Args:
            layers (int):  a list of intergers which represents the actual architecture o the feedforward netword
                            For example, layers = [2,2,1] would imply that our first input layers has two nodes,
                            our hidden one has two nodes, and our final output one has one nodes
            alpha (float, optional): the learning rate of our neural network. 
                                     this value is applied during the weight update phase           
                                     Defaults to 0.1.
        """
        # initialize the list of weight matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer
        # but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly inittialize weight matrix connection 
            # the number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] +1)

            # scale weight by dividing te square root of 
            # the number of nodes in the current layer
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a spcial case where the input
        # connections need a bias term the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    
    def sigmoid(self, x):
        """
        Compute and return the sigmoid activation value for a given input value

        Args:
            x (float): input value
        """
        return 1.0 / (1 + np.exp(-x))


    def sigmoid_deriv(self, x):
        """
        Compute the derivative of the sigmoid function ASSUMING
        that 'x' has already been passed through the sigmoid function

        Args:
            x (float): input value
        """
        return x * (1 - x)
    
    def __repr__(self):
        """
        Construct and return a string that represents the network architecture
        """
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))
    

    def fit(self, X, y, epochs=1000, display_update=100):
        """
        fit a model to the training dataset

        Args:
            X (matrix): the training dataset
            y (matrix): the corresponding class label for each entry in training dataset
            epochs (int, optional): the number of epochs that we'll train our network for
                                     Defaults to 1000.
            display_update (int, optional): the prameter simply controls how many epochs we'll 
                                            print training progress to our console.
                                             Defaults to 100.
        """
        # insert a column of 1's as the last entry in the feature matrix
        # this little trick allows us to treat the bias as 
        # a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over ech individual data point and train
            # our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update weight
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch +1 , loss
                ))

    def fit_partial(self, x, y):
        """
        First, Making  prediction on the data point -- Feed-forward phase
        Next, Computing the backpropagation PHASE
        Finally, Updating our weight matrix

        Args:
            x (matrix): the input training data
            y (matrix): the corresponding class labels for each entry in training dataset 
        """
        # construct our list of output activation for each layer
        # as our data point flows through the network;
        # the first activation is a special case -- it's just the 
        # input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and the
            # weight matrix -- this is called the "net input" to 
            # the current layer
            net = A[layer].dot(self.W[layer])

            # compute the "net output" is simply applying our
            # nonelinear activation to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our prediction (the final output activation in the activation list)
        # and the true taget value
        error = A[-1] - y

        # from here, we need to appy the chain rule and 
        # build our list of delta D; 
        # the first entry in the deltas list is simply the error of the
        #  output layer times the derivative of our activation 
        # function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # loop over the layers in reverse order ( ignorge the last
        #  two since we already have taken them into account)
        for layer in np.arange(len(A) -2, 0, -1):
            #the delta for the current layer is equal to the delta
            # of the previous layer dotted with the weight matrix
            # of the current layer, followed by  multiplying the delta 
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        
        # sine we loop over our layers in reverse order 
        # we need to reverse the deltas
        D = D[:: -1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" take place
            self.W[layer]+= -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        """
        Making prediction on the testing dataset

        Args:
            X (matrix): the input testing dataset
            add_bias (bool, optional): A boolean indicating whether weight  need add
                                        a column of 1's to input data to perform the bias trick.
                                        Defaults to True.
        """
        # intialize the output prediction as the input feature -- this value
        # will be (forward) propagated through the network to
        # obtain the final prediction
        p = np.atleast_2d(X)

        # check to see if the bias column should be added
        if add_bias:
            # insert a column of 1's as the last entry in the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computingt the output prediction is as simple as taking the
            # dot product between the current activation value 'pred'
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predict value
        return p


    def calculate_loss(self, X, targets):
        """
        This function will be used to calculate the
        loss across our entire training set.

        Args:
            X (matrix): the input data points
            targets (matrix): the corresponding class labes for each entry in data points
            
        Returns:
            float: the value of loss accoss whole input data
        """

        # ensure the targers matrix have at least two dimention
        targets = np.atleast_2d(targets)

        # make the predictions for the input data points
        predictions = self.predict(X, add_bias=False)

        # then compute the loss
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss