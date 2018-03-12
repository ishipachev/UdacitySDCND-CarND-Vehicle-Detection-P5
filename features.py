import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB':
        return img
    if (conv == 'RGB2YCrCb') or (conv == 'YCrCb'):
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV' or conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        return None


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys',
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()


# Define a function to compute color histogram features
def color_hist(image, nbins=16):
    channel1 = np.histogram(image[:, :, 0], bins=nbins, range=(0, 255))[0]
    channel2 = np.histogram(image[:, :, 1], bins=nbins, range=(0, 255))[0]
    channel3 = np.histogram(image[:, :, 2], bins=nbins, range=(0, 255))[0]
    hist = np.hstack((channel1, channel2, channel3))
    return hist


# Define a function to extract features from a list of images
def img_features(feature_image, hist_bins, orient, pix_per_cell,
                 cell_per_block, hog_channel, spatial_size):
    features = []
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    features.append(spatial_features)
    hist_features = color_hist(feature_image, nbins=hist_bins)
    features.append(hist_features)
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            channel_features = get_hog_features(feature_image[:, :, channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True)
            hog_features.append(channel_features)
        hog_features = np.ravel(hog_features)
    else:
        # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_LUV2RGB)
        # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
        hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    features.append(hog_features)
    return features


