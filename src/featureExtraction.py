from cgitb import grey
import enum
from unicodedata import name
import numpy as np
import pandas as pd
from scipy.stats import skew
import sys
from .Utils import resize_image


def ROI_color_feature_extraction(feature_vector, image):
    import cv2 as cv
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(grey_image, (15, 15), 0)
    th = cv.threshold(blurred_image, 200, 255,
                      cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    dilated_image = cv.erode(th, kernel, iterations=1)
    res = cv.bitwise_and(image, image, mask=dilated_image)

    for channel, color in zip(cv.split(res), ["Blue", "Green", "Red"]):
        histogram, bin_edges = np.histogram(channel, bins=64, range=(1, 256))
        for index, bin in enumerate(bin_edges[0:-1]):
            feature_vector[color+str(bin)] = histogram[index]
        feature_vector[color+"Mean"] = channel.mean()
        feature_vector[color+"Std"] = channel.std()
        feature_vector[color+"Skewness"] = skew(channel.reshape(-1))
    #cv.imshow("Dilated", dilated_image)
    #cv.imshow("After Mask", res)
    #cv.imshow("Dilated", dilated_image)
    #cv.imshow("After Mask", res)
    # cv.waitKey()
    # cv.waitKey()
    return feature_vector


def color_feature_extraction(feature_vector, image):
    import cv2 as cv
    # Feature: Add color distributions and color moments as attributes
    for channel, color in zip(cv.split(image), ["Blue", "Green", "Red"]):
        histogram, bin_edges = np.histogram(channel, bins=16, range=(0, 256))
        for index, bin in enumerate(bin_edges[0:-1]):
            feature_vector[color+str(bin)] = histogram[index]
        feature_vector[color+"Mean"] = channel.mean()
        feature_vector[color+"Std"] = channel.std()
        feature_vector[color+"Skewness"] = skew(channel.reshape(-1))
    return feature_vector

def binaryPatterns(feature_vector, image, numPoints, radius):
    from skimage import feature
    eps = 1e-7
    # print("Inside")
    lbp = feature.local_binary_pattern(
        image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(
        0, numPoints+3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum()+eps)
    for index, val in enumerate(hist):
        feature_vector["BP"+str(index)] = val
    return feature_vector

def canny_edge_feature_extraction(feature_vector, grey_image):
    import cv2 as cv
    sigma = 0.3
    median = np.median(grey_image)
    lower = int(max(0, (1.0-sigma)*median))
    upper = int(min(255, (1.0+sigma)*median))
    edge_canny = cv.Canny(grey_image, lower, upper)
    # plt.imshow(edge_canny)
    # plt.show()
    row_sums = np.sum(edge_canny, axis=0)
    column_sums = np.sum(edge_canny, axis=1)
    for index, row in enumerate(row_sums):
        feature_vector["Row"+str(index)] = row
    for index, column in enumerate(column_sums):
        feature_vector["Column"+str(index)] = column
    return feature_vector

def orb_feature_extraction(feature_vector, image, size):
    import cv2 as cv
    alg = cv.ORB_create(nfeatures=5000)
    kps = alg.detect(image)
    n = 500
    kps = sorted(kps, key=lambda x: -x.response)[:n]
    keypoint_matrix = np.zeros(size)

    # compute descriptor values from keypoints (128 per keypoint)
    kps, dsc = alg.compute(image, kps)
    
    for point in kps:
        x, y = point.pt
        keypoint_matrix[round(x)][round(y)] = point.size

    #img2 = cv.drawKeypoints(image, kps, None, color=(0,255,0), flags=0)
    #cv.imshow("Keypoints",img2)
    #plt.show()
    #cv.waitKey(0)
    try:
        vector = dsc.reshape(-1)
    except:
        vector = np.zeros(n*32)

    if vector.size < (n*32):
       # It can happen that there are simply not enough keypoints in an image,
       # in which case you can choose to fill the missing vector values with zeroes
        vector = np.concatenate([vector, np.zeros(n*32 - vector.size)])
    for i in range(len(vector)):
        feature_vector["ORB"+str(i)] = vector[i]
    return feature_vector


def traditional_feature_extraction(path, kernels, size=(640, 350)):
    import matplotlib.pyplot as plt
    import cv2 as cv
    # print(path)
    img = cv.imread(path)
    image = resize_image(img, size)
    original = image.copy()
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    name_df = {}
    name_df["Name"] = path
    feature_vector = {}
    #feature_vector = ROI_color_feature_extraction(feature_vector, image)
    #feature_vector = color_feature_extraction(feature_vector, image)
    #feature_vector = canny_edge_feature_extraction(feature_vector, grey_image)
    feature_vector = binaryPatterns(feature_vector, grey_image, 24, 8)
    #feature_vector = orb_feature_extraction(feature_vector, image, size)
    name_df = pd.DataFrame([name_df])
    df1 = pd.DataFrame([feature_vector])
    returndf = name_df.join(df1)
    return returndf


def dnn_feature_exctration(file, model, image_size):
    from tensorflow.keras.utils import load_img
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from PIL import Image
    img = load_img(file, target_size=(image_size, image_size))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, image_size, image_size, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return pd.DataFrame(features)
