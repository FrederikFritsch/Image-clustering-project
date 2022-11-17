import numpy as np
#import cv2 as cv
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import sys
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from clusteringAlgorithms import *
from featureExtraction import *
from filterCreation import *
from supportFunctions import *


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        data_dir = str(args[0])
        normalization_method = str(args[1])
        image_size = (int(args[2]), int(args[2]))
        pca_variance = float(args[3])
        show_plots = int(args[4])
        min_clusters = int(args[5])
        max_clusters = int(args[6])
        debug = int(args[7])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()
        
    #image_size = (320, 175)
    take_time = True
    base_dir = os.getcwd()
    full_data_dir_path = base_dir + data_dir
    
    all_image_paths = get_image_paths(full_data_dir_path) #Get image paths in data directory
    gabor_filters = create_gabor_filters() # Creates list of Gabor filters
    

    if take_time: starttime = time.time() #Start time
    # Feature Extraction
    dataframe_list = []
    for path in all_image_paths:
        dataframe = traditional_feature_extraction(path, gabor_filters, image_size)
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list)
    
    if take_time: endtime = time.time() #Stop time
    if take_time: print(f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
    

    # Feature normalization
    features = df.columns[1:]
    if debug: print(features)
    if normalization_method == "normalize":
        df[features] = normalize(df[features])
    else:
        scaler = StandardScaler()
        scaler.fit(df[features])
        df[features] = scaler.transform(df[features])

    # Dimensionality reduction
    pca = PCA(pca_variance)
    pca.fit(df[features])
    scores_pca = pca.transform(df[features])
    if debug: print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering
    sse, score, silhouette_coefficients, labels = perform_KMeans(scores_pca, min_clusters, max_clusters)


    # Cluster number evaluation

    kl = KneeLocator(range(min_clusters, max_clusters+1), sse, curve="convex", direction="decreasing")
    if debug: print(kl.elbow)
    if kl.elbow:
        n_clusters = kl.elbow
    else:
        n_clusters = np.argmax(silhouette_coefficients)+min_clusters
    if debug: print(f"Silhouette coefficient: {n_clusters} clusters return best results")


    # Only Plotting below this line
    if show_plots:
        X = TSNE(n_components=2, perplexity=40).fit_transform(scores_pca)
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
            alpha=1
        )
        plt.show()



        #Image merging and showing/storing
    
        for cluster_number in range(n_clusters):
            cluster = tsne_df.loc[tsne_df["ClusterID"]==cluster_number]
            image_list = []
            for image_path in cluster["Image Name"]:
                image_list.append(image_path)

            column_number = int(np.ceil(np.sqrt(len(image_list))))
            if len(image_list):
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