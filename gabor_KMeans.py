import numpy as np
import cv2 as cv
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from scipy.stats import skew
import os
import time
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image


def combine_images(columns, space, images):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    return background
    #background.save('image.png')

def traditional_feature_extraction(path, size = 244, kernelsize = (10, 20), thetarotations = 4, sigmas = (1,3), lamdas = (np.pi /2, np.pi), gammas = (0.5, 0.05)):
    image = cv.imread(path)
    image = cv.resize(image, (size, size), interpolation= cv.INTER_LINEAR)   

    df = pd.DataFrame()
    
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
    kernels = []
    gabor_features = {}

    for ksize in kernelsize:                              # Kernel size
        for theta in range(thetarotations):               # number of rotations
            theta = theta / 4. * np.pi
            for sigma in sigmas:                        # SIGMA with 1 and 3
                for lamda in lamdas:         # range of wavelengths
                    for gamma in gammas:           # GAMMA values of 0.05 and 0.5
                        gabor_label = 'Gabor' + str(num)
                        phi = 0
                        #CREATE Kernel and APPLY Filter
                        kernel = cv.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv.CV_32F)
                        kernels.append(kernel)
                        fimg = cv.filter2D(image2, cv.CV_8UC3, kernel)
                        filtered_image = np.array(fimg.reshape(-1))

                        #Calculate mean and std                    
                        mean = sum(filtered_image)/len(filtered_image)
                        std = np.std(filtered_image)

                        #Add mean and std to feature vector
                        gabor_mean = gabor_label + "Mean"
                        gabor_std = gabor_label + "Std"

                        gabor_features[gabor_mean] = mean
                        gabor_features[gabor_std] = std
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


def get_image_paths(full_data_dir_path):
    all_paths = []
    for index, directories in enumerate(os.walk(full_data_dir_path)):
        for sample in directories[2]:
            if sample.endswith('.png'):
                full_path = directories[0] + "/" + sample
                all_paths.append(full_path)
    return all_paths

if __name__ == "__main__":
    take_time = True
    base_dir = os.getcwd()
    data_dir = "/Image_Data/"
    full_data_dir_path = base_dir + data_dir
    
    # ------ GET ALL IMAGE PATHS IN DATA DIRECTORY --------
    all_image_paths = get_image_paths(full_data_dir_path)

    
    image_size = 244
    dataframe_list = []
    if take_time: starttime = time.time()

    # ------- APPLY TRADITIONAL FEATURE EXTRACTION METHODS -----------

    for path in all_image_paths:
        dataframe = traditional_feature_extraction(path, image_size)
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list)

    if take_time: endtime = time.time()
    if take_time: print(f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
    

    # -------- STANDARDIZE FEATURE DATA (Z-TRANSFORM) --------------
    features = df.columns[1:]
    print(features)
    scaler = StandardScaler()
    scaler.fit(df[features])
    df[features] = scaler.transform(df[features])
    #print(df)

    # -------- APPLY PCA FEATURES --------------
    pca = PCA(0.95)
    pca.fit(df[features])
    print(f"Explained components: {pca.explained_variance_ratio_}")

    scores_pca = pca.transform(df[features])
    min_clusters = 2
    max_clusters = 5
    sse = []
    silhouette_coefficients = []
    labels = []

    # --------- CALCULATE K-MEANS CLUSTERS ------------
    for nr_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(init = "random", n_clusters = nr_clusters, n_init = 10, max_iter=300, random_state = 22)
        kmeans.fit(scores_pca)
        sse.append(kmeans.inertia_)
        score = silhouette_score(scores_pca, kmeans.labels_)
        silhouette_coefficients.append(score)
        labels.append(kmeans.labels_)


    # ---------- EVALUATE CLUSTER SIZES --------------
    kl = KneeLocator(range(min_clusters, max_clusters+1), sse, curve="convex", direction="decreasing")
    print(kl.elbow)
    n_clusters = np.argmax(silhouette_coefficients)+min_clusters
    print(f"Silhouette coefficient: {n_clusters} clusters return best results")


    # ----------- CALCULATE TSNE FOR PLOTTING ---------
    X = TSNE(n_components=2, perplexity=4).fit_transform(scores_pca)
    tsne_df = pd.DataFrame()
    cluster_labels = pd.Series(labels[np.argmax(silhouette_coefficients)])
    tsne_df["Image Name"] = df["Name"]
    tsne_df['ClusterID'] = cluster_labels.values
    tsne_df["X_tsne"]  = X[:, 0]
    tsne_df["Y_tsne"] = X[:, 1]
    
    print(tsne_df)

    fig, axes = plt.subplots(2, 1)
    plt.style.use("fivethirtyeight")
    axes[0].plot(range(min_clusters, max_clusters+1), sse)
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("SSE")
    axes[1].plot(range(min_clusters, max_clusters+1), silhouette_coefficients)
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Coefficient")
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="X_tsne", y="Y_tsne",
        hue="ClusterID",
        palette=sns.color_palette("hls", 10),
        data=tsne_df,
        legend="full",
        alpha=0.9
    )
    #plt.show()


# ---------- STORE IMAGE PATHS OF EACH CLUSTER IN LISTS ------------
    

    for cluster_number in range(n_clusters):
        cluster = tsne_df.loc[tsne_df["ClusterID"]==cluster_number]
        image_list = []
        for image_path in cluster["Image Name"]:
            image_list.append(image_path)

        column_number = int(np.ceil(np.sqrt(len(image_list))))
        
        merged_image = combine_images(columns=column_number, space=10, images=image_list)
        merged_image.show()

#Second set - NEED TO DO EDGE DETECTION FOR RGB

#Canny edge
#edge_canny = cv.Canny(image2, 100, 200)
#edge_canny = edge_canny.reshape(-1)
#df['Canny Edge'] = edge_canny
#
##Roberts edge
#edge_roberts = roberts(image)
#edge_roberts = edge_roberts.reshape(-1)
#df['Roberts'] = edge_roberts
#
##Sobel edge
#edge_sobel = sobel(image)
#edge_sobel = edge_sobel.reshape(-1)
#df['Sobel'] = edge_sobel
#
##Scharr edge
#edge_scharr = scharr(image)
#edge_scharr = edge_scharr.reshape(-1)
#df['Scharr'] = edge_scharr
#
##Prewitt edge
#edge_prewitt = prewitt(image)
#edge_prewitt = edge_prewitt.reshape(-1)
#df['Prewitt'] = edge_prewitt

#print(df.iloc[:,25])
#cv.imshow("Filtered Image23", (df.iloc[:,23].values).reshape(grey_image.shape))
#cv.imshow("Filtered Image22", (df.iloc[:,22].values).reshape(grey_image.shape))
#print(df.head())
#cv.waitKey(0)