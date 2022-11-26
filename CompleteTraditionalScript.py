import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import sys
#from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import seaborn as sns
from clusteringAlgorithms import *
from featureExtraction import *
from Utils import *
import datetime


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        data_dir = str(args[0])
        normalization_method = str(args[1])
        image_size = (int(args[2]), int(args[2]))
        pca_variance = float(args[3])
        show_plots = False
        min_clusters = int(args[5])
        max_clusters = int(args[6])
        debug = False
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()
    
    take_time = True
    base_dir = os.getcwd()
    #full_data_dir_path = base_dir + data_dir
    full_data_dir_path = data_dir

    all_image_paths = get_image_paths(full_data_dir_path) #Get image paths in data directory
    gabor_filters = create_gabor_filters() # Creates list of Gabor filters
    

    if take_time: starttime = time.time() #Start time
    # Feature Extraction
    dataframe_list = []
    for path in all_image_paths:
        dataframe = traditional_feature_extraction(path, gabor_filters, image_size)
        dataframe_list.append(dataframe)
    df = pd.concat(dataframe_list, ignore_index=True)
    #print(df.info())
    #print(df.head())
    #print(df.describe())
    if take_time: endtime = time.time() #Stop time
    if take_time: print(f"Time elapsed to extract features of {len(all_image_paths)} Images: {endtime-starttime}")
    

    # Feature normalization
    features_df = df.loc[:, df.columns[1:]]
    image_names_df = df.loc[:, ['Name']]
    
    #print(image_names_df)
    if debug: print(features_df.describe())
    if normalization_method == "normalize":
        print("Normalizing")
        features_df = normalize(features_df)
    else:
        scaler = StandardScaler()
        print("Standardizing")
        #scaler.fit(features_df)
        features_df = scaler.fit_transform(features_df)
    
    # Dimensionality reduction
    pca = PCA(pca_variance)

    features_pca_df = pca.fit_transform(features_df)

    print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering
    sse, score, silhouette_coefficients, labels = perform_KMeans(features_pca_df, min_clusters, max_clusters)


    # Cluster number evaluation

    #kl = KneeLocator(range(min_clusters, max_clusters+1), sse, curve="convex", direction="decreasing")
    #if debug: print(f"Elbow Method returns: {kl.elbow} Clusters")
    #if kl.elbow:
    #    n_clusters = kl.elbow + 1
    #else:
    n_clusters = np.argmax(silhouette_coefficients) + min_clusters
    print(f"Silhouette coefficient: {n_clusters} clusters return best results")
    
    lables_index = np.argmax(silhouette_coefficients)
    cluster_labels = pd.DataFrame(labels[lables_index], columns=["Cluster"])

    results_df = pd.concat([image_names_df, cluster_labels], axis=1)

    # Only Plotting below this line
    if show_plots:
        X = TSNE(n_components=2, perplexity=5).fit_transform(features_pca_df)
        tsne_df = pd.DataFrame()
        #cluster_labels = pd.Series(labels[np.argmax(silhouette_coefficients)])
        tsne_df["Image Name"] = df["Name"]
        tsne_df['ClusterID'] = cluster_labels.values
        tsne_df["X_tsne"]  = X[:, 0]
        tsne_df["Y_tsne"] = X[:, 1]

        #print(tsne_df)
        fig, axes = plt.subplots(2, 1)
        plt.style.use("fivethirtyeight")
        axes[0].plot(range(min_clusters, max_clusters+1), sse)
        axes[0].set_xlabel("Number of Clusters")
        axes[0].set_ylabel("SSE")
        axes[1].plot(range(min_clusters, max_clusters+1), silhouette_coefficients)
        axes[1].set_xlabel("Number of Clusters")
        axes[1].set_ylabel("Silhouette Coefficient")

        plt.figure(figsize=(16,10))
        #sns.scatterplot(
        #    x="X_tsne", y="Y_tsne",
        #    hue="ClusterID",
        #    palette=sns.color_palette("hls", 10),
        #    data=tsne_df,
        #    legend="full",
        #    alpha=1
        #)
        plt.show()



    #Image merging and showing/storing
    date = datetime.datetime.now()
    dir_name = date.strftime("%c")#.replace(":", "")
    os.mkdir(base_dir+"/Results/Traditional/"+dir_name)
    results_path = base_dir+"/Results/Traditional/"+dir_name+"/"
    os.chdir(results_path)
    for cluster_number in range(n_clusters):
        cluster = results_df.loc[results_df["Cluster"]==cluster_number]
        image_list = []
        for image_path in cluster["Name"]:
            image_list.append(image_path)
        column_number = int(np.ceil(np.sqrt(len(image_list))))
        if len(image_list) > 0:

            merged_image = combine_images(columns=column_number, space=10, images=image_list)
            #merged_image.save(str(cluster_number)+".png")
            merged_image.show()
    os.chdir(base_dir)

