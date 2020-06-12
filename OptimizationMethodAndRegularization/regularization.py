# import the necessary libraries
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreproccesor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import  argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to the input dataset")
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor
# load the dataset from disk
# and reshape the data matrix
sp = SimplePreproccesor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])

(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as intergers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into the training and testing splits using 75% of 
# the whole data for training and the remaining 25% for testing
(train_X, test_X, train_Y, test_Y)= train_test_split(data, labels,
    test_size=0.25, random_state=5)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    #train a SGD classifier using a softmax loss function and
    # the specified regularization function for 10 epochs
    print("[INFO] training model with {} penalty". format(r))
    model = SGDClassifier( loss="log", penalty=r, max_iter= 10,
        learning_rate="constant", eta0= 0.01, random_state=42)
    model.fit(train_X, train_Y)

    # evaluate the classifier
    acc = model.score(test_X, test_Y)
    print("[INFO] {} penalty accuracy: {:.2f}%".format(r, acc * 100))
