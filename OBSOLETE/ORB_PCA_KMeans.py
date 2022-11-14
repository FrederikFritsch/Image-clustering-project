#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

#from clustimage import Clustimage

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

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

path = "MY_data/predict/*.jpeg"
path2 = "MY_data/train/*/*.jpeg"

images = []
try:
    images = [cv.imread(file) for file in glob.glob(path)]
    print(f"Imported: {len(images)} images")
except:
    print("Error when importing images.")


# ----------- ORB KEYPOINT AND DESCRIPTOR EXTRACTION ______-
# Initiate ORB detector
orb = cv.ORB_create(nfeatures = 1000)
#
#desc_list = []
#kp_list = []
#for image in images:
#    kp, dsc = orb.detectAndCompute(image, None)
#    desc_list.append(dsc)
#    kp_list.append(kp)
#
#for desc in desc_list:
#    print(desc.size)


img1 = cv.imread('img_371.jpeg', 1)          # queryImage
img2 = cv.imread('img_361.jpeg', 1) # trainImage

# Initiate ORB detector

METHOD = "sift"
if METHOD == "orb":
    detector = cv.ORB_create(nfeatures = 1000)
else:
    detector = cv.SIFT_create(nfeatures = 1000)

# find the keypoints and descriptors with detector

kp1, des1 = detector.detectAndCompute(img1,None)
kp2, des2 = detector.detectAndCompute(img2,None)

img4 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img5 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
cv.imshow("First", img4)
cv.imshow("second", img5)

if METHOD == "sift":
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
else:
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level = 2) #2

search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for i, pair in enumerate(matches):
    try:
        m,n = pair
        if m.distance < 0.7*n.distance:
            good.append(m)
    except ValueError:
        pass

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

#feature Extraction
bf = cv.BFMatcher(cv.NORM_L2, crossCheck = True)
#
#
## find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2, None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, img2)

# draw only keypoints location,not size and orientation
#img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

cv.imshow("ORB", img3)
cv.waitKey(0)