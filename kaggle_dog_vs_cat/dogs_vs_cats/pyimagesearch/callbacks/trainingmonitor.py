"""
    To Serialize the loss and accuracy for both the training and 
    validation set to disk, followed construct a plot of the data
"""
# import the necessary libraries
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath, startAt=0):
        """
        Store the ouput path for the figure, the path to the JSON
        serialized file and the starting epoch

        Args:
            figPath (string): the  path to the output figure that we can use to visualize loss 
                                and accuracy over time
            jsonPath (string): the path to the json serialized file
            startAt (int, optional): the starring epoch that training is resumed at when using
                                    ctrl + c training. Defaults to 0.
        """
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
    
    def on_train_begin(self, logs={}): 
        """
        this function is called once when the training process starts

        Args:
            logs (dict, optional): the history of loss and accuracy. Defaults to {}.
        """
        # initialize the history dictionary
        self.H = {}
        # if the JSON history path exist, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and trim
                    # any entries that are past of starting epoch
                    for k  in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
        
    def on_epoch_end(self, epoch, logs={}):
        """
        this function is called once when the training process finish

        Args:
            epoch (int): the current epoch
            logs (dict, optional): the history of loss and accuracy. Defaults to {}.
        """
        # loop over the logs and update the loss , accuracy ,etc
        # for the entire the process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        
        # check to see if th training history should be 
        # serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H ,default = lambda x: str(x)))
            f.close()
        
        # ensure  at least two epochs have passed before plotting
        # epoch starting at zeros
        if len(self.H["loss"]) > 1:
            # plot the trainng loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title("The Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            # save plot
            plt.savefig(self.figPath)
            plt.close()






