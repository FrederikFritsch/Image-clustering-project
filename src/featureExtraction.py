import enum
import numpy as np
import pandas as pd
from scipy.stats import skew


def traditional_feature_extraction(path, kernels, size=(320, 175)):
    import matplotlib.pyplot as plt
    import cv2 as cv
    # print(path)
    img = cv.imread(path)

    image = cv.resize(img, size, interpolation=cv.INTER_LINEAR)
    #image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # print(image.shape)
    color_distributions = {}
    color_distributions["Name"] = path

    # Feature 1: Add color distributions as attributes
    for channel, color in zip(cv.split(image), ["Blue", "Green", "Red"]):
        histogram, bin_edges = np.histogram(channel, bins=16, range=(0, 256))
        for index, bin in enumerate(bin_edges[0:-1]):
            color_distributions[color+str(bin)] = histogram[index]
        color_distributions[color+"Mean"] = channel.mean()
        color_distributions[color+"Std"] = channel.std()
        color_distributions[color+"Skewness"] = skew(channel.reshape(-1))

    # Convert to grayscale for gabor filters
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #image2 = grey_image.reshape(-1)
    #num = 1
    #gabor_features = {}
#
    # for kernel in kernels:
    #    gabor_label = 'Gabor' + str(num)
    #    fimg = cv.filter2D(image2, cv.CV_8UC3, kernel)
    #    filtered_image = np.array(fimg.reshape(-1))
#
    #    #Calculate mean and std
    #    mean = sum(filtered_image)/len(filtered_image)
    #    std = np.std(filtered_image)
#
    #    #Add mean and std to feature vector
    #    gabor_mean_label = gabor_label + "Mean"
    #    gabor_std_label = gabor_label + "Std"
#
    #    gabor_features[gabor_mean_label] = mean
    #    gabor_features[gabor_std_label] = std
    #    num += 1
#
    #    #df[gabor_label] = filtered_image
    #    #print(gabor_label, mean, std)
    #    kernel_resized = cv.resize(kernel, (400, 400))
    #    #cv.imshow("Kernel: Theta " + str(theta) +" Sigma "+ str(sigma) +" Lamda "+ str(lamda) +" Gamma "+ str(gamma), kernel_resized)
    #    #cv.imshow("Kernel", kernel_resized)
    #    #cv.imshow("Original img", image)
    #    #cv.imshow("Filtered", filtered_image.reshape(grey_image.shape))
    #    #cv.waitKey(0)

    # Canny edge
    edge_features = {}
    #edge_canny = cv.Canny(grey_image, 300,500)
    # plt.imshow(edge_canny)
    # plt.show()
    #row_sums = np.sum(edge_canny, axis = 0)
    #column_sums = np.sum(edge_canny, axis = 1)
    # for index, row in enumerate(row_sums):
    #   edge_features["Row"+str(index)] = row
    # for index, column in enumerate(column_sums):
    #   edge_features["Column"+str(index)] = column

    alg = cv.ORB_create(nfeatures=5000)
    kps = alg.detect(image)
    n = 50
    kps = sorted(kps, key=lambda x: -x.response)[:n]
    
    # compute descriptor values from keypoints (128 per keypoint)
    kps, dsc = alg.compute(image, kps)

    #print(f"KPS: {kps}")
    #print(f"DSC: {dsc}")
    img2 = cv.drawKeypoints(image, kps, None, color=(0,255,0), flags=0)
    cv.imshow("Keypoints",img2)
    cv.waitKey(0)
    try:
        vector = dsc.reshape(-1)
    except:
        vector = np.zeros(n*32)

    if vector.size < (n*32):
       # It can happen that there are simply not enough keypoints in an image,
       # in which case you can choose to fill the missing vector values with zeroes
        vector = np.concatenate([vector, np.zeros(n*32 - vector.size)])
    # print(vector)
    orb_descriptors = {}
    for i in range(len(vector)):
        orb_descriptors["ORB"+str(i)] = vector[i]

    df1 = pd.DataFrame([color_distributions])
    #df2 = pd.DataFrame([gabor_features])
    df3 = pd.DataFrame([edge_features])
    df4 = pd.DataFrame([orb_descriptors])
    returndf = df1.join(df3).join(df4)
    return returndf


def dnn_feature_exctration(file, model):
    from keras.utils import load_img
    from keras.applications.vgg16 import preprocess_input

    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return pd.DataFrame(features)
