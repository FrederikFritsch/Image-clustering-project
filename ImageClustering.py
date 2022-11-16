import numpy as np
#import cv2 as cv
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
from clusteringAlgorithms import perform_KMeans
from featureExtraction import traditional_feature_extraction
from filterCreation import create_gabor_filters

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
    NORMALIZE = True
    
    # ------ GET ALL IMAGE PATHS IN DATA DIRECTORY --------
    all_image_paths = get_image_paths(full_data_dir_path)

    
    image_size = 244
    dataframe_list = []
    if take_time: starttime = time.time()

    # ------- APPLY TRADITIONAL FEATURE EXTRACTION METHODS -----------
    gabor_filters = create_gabor_filters()

    for path in all_image_paths:
        dataframe = traditional_feature_extraction(path, gabor_filters, image_size)
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list)

    if take_time: endtime = time.time()
    if take_time: print(f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
    

    # -------- STANDARDIZE FEATURE DATA (Z-TRANSFORM) --------------
    features = df.columns[1:]
    #print(features)
    if NORMALIZE:
        df[features] = normalize(df[features])
        
    else:
        scaler = StandardScaler()
        scaler.fit(df[features])
        df[features] = scaler.transform(df[features])
    print(df)

    # -------- APPLY PCA FEATURES --------------
    pca = PCA(0.95)
    pca.fit(df[features])
    print(f"Explained components: {pca.explained_variance_ratio_}")

    scores_pca = pca.transform(df[features])
    min_clusters = 3
    max_clusters = 20


    # ---- CALCULATE KMeans Clusters
    sse, score, silhouette_coefficients, labels = perform_KMeans(scores_pca, min_clusters, max_clusters)


    # ---------- EVALUATE CLUSTER SIZES --------------

    kl = KneeLocator(range(min_clusters, max_clusters+1), sse, curve="convex", direction="decreasing")
    print(kl.elbow)
    if kl.elbow:
        n_clusters = kl.elbow
    else:
        n_clusters = np.argmax(silhouette_coefficients)+min_clusters
    print(f"Silhouette coefficient: {n_clusters} clusters return best results")


    # ----------- CALCULATE TSNE FOR PLOTTING ---------
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
        alpha=0.9
    )
    plt.show()


# ---------- STORE IMAGE PATHS OF EACH CLUSTER IN LISTS ------------
    

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