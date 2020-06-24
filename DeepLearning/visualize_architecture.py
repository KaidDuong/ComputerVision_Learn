# import the necessary libraries
from pyimagesearch.neural_network.cnn.lenet import LeNet
from keras.utils import plot_model
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help="path to the network architecture visualize graph")
args = vars(ap.parse_args())

# initialize the LeNet architecture and then write network visualize graph to the disk
model = LeNet.build(width=28, height=28, depth=1, classes=10)
plot_model(model, to_file=args["output"],
    show_layer_names=True, show_shapes=True )
