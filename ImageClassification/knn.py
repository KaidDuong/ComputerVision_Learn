# Import the necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreproccesor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
         help="path to input dataset")
ap.add_argument('-k',"--neighbors",type=int, default =1, 
        help="# of nearest neighbors for classification ")
ap.add_argument("-j","--jobs",type=int,  default=-1,
         help="# of jobs of Knn distance( -1 uses all available cores)")
args = vars(ap.parse_args())

# Grap the list images that we'll be describe
print("[INFO]loading image ...")
imagePaths = list(paths.list_images(args["dataset"]))   

# Initial the image processor
# load the dataset from disk and reshape the data matrix

sp = SimplePreproccesor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some info on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))

# encode the labels as intergers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing split using 75% of
# the data for training and remaining 25% for testing
(trainX, testX, trainY,testY) = train_test_split(data, labels, 
                                test_size=.25, random_state=42)

# train and evaluate a  knn classifier on the raw pixel intensities 
knn = KNeighborsClassifier(n_neighbors=args["neighbors"],
                            n_jobs= args["jobs"])

knn.fit(trainX, trainY)
pred_Y = knn.predict(testX)
print(classification_report(testY, pred_Y,
        target_names=le.classes_))