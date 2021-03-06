import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
from sklearn.model_selection import train_test_split
from features import convert_color
from features import img_features
from sklearn.utils import shuffle

#just some stupid test
#and another one
#and another one

def read_image_folders():
    cars = []
    cars.append(glob.glob('vehicles\GTI_Far\*.png'))
    cars.append(glob.glob('vehicles\GTI_Left\*.png'))
    cars.append(glob.glob('vehicles\GTI_MIddleClose\*.png'))
    cars.append(glob.glob('vehicles\GTI_Right\*.png'))
    cars.append(glob.glob('vehicles\KITTI_extracted\*.png'))
    cars = [item for sublist in cars for item in sublist]

    notcars = []
    notcars.append(glob.glob('non-vehicles/Extras/*.png'))
    notcars.append(glob.glob('non-vehicles/GTI/*.png'))
    notcars = [item for sublist in notcars for item in sublist]

    return cars, notcars


def extract_features(imgs, colorspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)  # Read in each imageone by one
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, colorspace)
        file_features = img_features(feature_image, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel, spatial_size)
        features.append(np.concatenate(file_features))
    return features


cars, notcars = read_image_folders()

print(len(cars), len(notcars))

cars = shuffle(cars)
notcars = shuffle(notcars)

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
# sample_size = 2500
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9   # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 2  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32     # Number of histogram bins


t = time.time()
car_features = extract_features(cars, colorspace=colorspace,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, colorspace=colorspace,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

## Saving model in .pickle file
svc_dict = {"svc": svc,
            "scaler": X_scaler,
            "orient": orient,
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,
            "colorspace": colorspace,
            "hog_channel": hog_channel,
            "spatial_size": spatial_size,
            "hist_bins": hist_bins
            }

model_path = 'output/svc_model.p'

with open(model_path, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(svc_dict, f, pickle.HIGHEST_PROTOCOL)

# Testing correctness of saving model
with open(model_path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)
    svc_load = data["svc"]

n_predict = 10
print('Printing accuracy of loaded model')
print('My SVC predicts: ', svc_load.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
