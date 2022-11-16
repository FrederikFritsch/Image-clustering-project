import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import skew


def traditional_feature_extraction(path, kernels, size = 244):
    image = cv.imread(path)
    image = cv.resize(image, (size, size), interpolation= cv.INTER_LINEAR)   
    
    color_distributions = {}
    color_distributions["Name"] = path

    #Feature 1: Add color distributions as attributes
    for channel, color in zip(cv.split(image), ["Blue", "Green", "Red"]):
        color_distributions[color+"Mean"] = channel.mean()
        color_distributions[color+"Std"] = channel.std()
        color_distributions[color+"Skewness"] = skew(channel.reshape(-1))


    #Convert to grayscale for gabor filters
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    image2 = grey_image.reshape(-1)
    num = 1
    gabor_features = {}

    for kernel in kernels:
        gabor_label = 'Gabor' + str(num)
        fimg = cv.filter2D(image2, cv.CV_8UC3, kernel)
        filtered_image = np.array(fimg.reshape(-1))

        #Calculate mean and std                    
        mean = sum(filtered_image)/len(filtered_image)
        std = np.std(filtered_image)

        #Add mean and std to feature vector
        gabor_mean_label = gabor_label + "Mean"
        gabor_std_label = gabor_label + "Std"

        gabor_features[gabor_mean_label] = mean
        gabor_features[gabor_std_label] = std
        num += 1

        #df[gabor_label] = filtered_image
        #print(gabor_label, mean, std)
        #kernel_resized = cv.resize(kernel, (400, 400))
        #cv.imshow("Kernel: Theta " + str(theta) +" Sigma "+ str(sigma) +" Lamda "+ str(lamda) +" Gamma "+ str(gamma), kernel_resized)
        #cv.imshow("Kernel", kernel_resized)
        #cv.imshow("Original img", image)
        #cv.imshow("Filtered", filtered_image.reshape(grey_image.shape))
        #cv.waitKey(0)
    
    df1 = pd.DataFrame([color_distributions])
    df2 = pd.DataFrame([gabor_features])
    returndf = df1.join(df2)
    return returndf