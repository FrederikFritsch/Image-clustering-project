#from keras.preprocessing.image import load_img 
#from keras.preprocessing.image import img_to_array 
#from keras.applications.vgg16 import preprocess_input
#
## models 
#from keras.applications.vgg16 import VGG16 
#from keras.models import Model
#
## clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import glob
import cv2 as cv
from collections import Counter



img1 = cv.imread('img_01.jpeg',0)
img2 = cv.imread('img_191.jpeg',0)

path = "MY_data/predict/*.jpeg"
path2 = "MY_data/train/*/*.jpeg"
images = []
try:
    images = [cv.imread(file) for file in glob.glob(path2)]
    print(f"Imported: {len(images)} images")
except:
    print("Error when importing images.")

flatten_images = []
#cv.imshow("before", images[1])
for i, image in enumerate(images):
    try:
        flatten_images.append(cv.resize(image, (144, 144), interpolation= cv.INTER_LINEAR).flatten())
    except:
        print(f"Error on file {i}")
#cv.imshow("after", images[1])
scalar = StandardScaler()
scaled_images = scalar.fit_transform(flatten_images)

pca = PCA(100)

pca.fit(scaled_images)
sum = 0.0
for variance in pca.explained_variance_ratio_:
    tmp = float("{:.4f}".format(variance))
    sum += tmp
print(f"Total: {sum}")
#img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

scores_pca = pca.transform(scaled_images)

#wcss = []
#for i in range(1, 21):
#    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state =42)
#    kmeans_pca.fit(scores_pca)
#    wcss.append(kmeans_pca.inertia_)
#

#plt.figure(figsize=(10,8))
#plt.plot(range(1,21), wcss, marker ='o', linestyle='--')
#plt.xlabel("Number of Clusters")
#plt.ylabel("K-Means with PCA Clustering")
#plt.show()

kmeans_pca = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)

clusters = kmeans_pca.labels_
cluster_count = Counter(clusters)
print(cluster_count)

for clust in range(10):
    cluster_images = [] 
    for i, cluster_nr in enumerate(clusters):
        if cluster_nr == clust:
            #cv.imshow(f"Image {i}", images[i])
            cluster_images.append(cv.resize(images[i], (144, 144), interpolation= cv.INTER_LINEAR))


    image_rows = [] #np.concatenate((cluster_images[0], cluster_images[1]))
    for i, image in enumerate(cluster_images):
        if i%15 == 0:
            #print(i)
            try:
                for j in range(1,15):
                   cluster_images[i] = np.concatenate((cluster_images[i], cluster_images[i+j]), axis = 1)
                image_rows.append(cluster_images[i])
            except:
                None
            #cluster_images[0] = np.concatenate((cluster_images[0], cluster_images[i]), axis=0)

    #for i, row in enumerate(image_rows):
    #    cv.imshow(f"Cluster {i}", image_rows[i])

    image_grid = []

    for i in range(len(image_rows)-1):
        image_rows[0] = np.concatenate((image_rows[0], image_rows[i+1]), axis = 0)
    #cv.imshow(f"Cluster one", cluster_images[0])
    cv.imshow(f"Cluster {clust}", image_rows[0])
cv.waitKey(0)
## Initiate ORB detector
#orb = cv.ORB_create()
#
##feature Extraction
#bf = cv.BFMatcher(cv.NORM_L2, crossCheck = True)
#
#
## find the keypoints with ORB
#kp1 = orb.detect(img1,None)
#kp2 = orb.detect(img2, None)
## compute the descriptors with ORB
#kp1, des1 = orb.compute(img1, kp1)
#kp2, des2 = orb.compute(img2, kp2)
#
#matches = bf.match(des1, des2)
#matches = sorted(matches, key = lambda x:x.distance)
#
#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, img2)
#
## draw only keypoints location,not size and orientation
##img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
#
##cv.imshow("ORB", img3)
##cv.waitKey(0)