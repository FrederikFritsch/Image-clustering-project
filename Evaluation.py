import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import seaborn as sns
from src.Utils import *
import datetime


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        filepath = str(args[0])
        dir = str(args[1])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()

    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(e)
        quit()
    print(df)

    
    #n_clusters = df.max()["Cluster"]+1
    n_clusters = len(df["Cluster"].unique())
    print(n_clusters)

    date = datetime.datetime.now()
    dir_name = date.strftime("%c")#.replace(":", "")
    basedir = os.getcwd()+"/"
    os.chdir(basedir+dir)
    print(basedir+dir)
    for cluster_number in range(n_clusters):
        cluster = df.loc[df["Cluster"]==cluster_number]
        image_list = []
        for image_path in cluster["Name"]:
            image_list.append(basedir+image_path)
        column_number = int(np.ceil(np.sqrt(len(image_list))))
        if len(image_list) > 0:
            print("Merging images")
            merged_image = combine_images(columns=column_number, space=10, images=image_list)
            merged_image.save(str(cluster_number)+".png")
            merged_image.show()
    #print(n_unique)
    #print(f"Silhouette coefficient: {n_clusters} clusters return best results")
    
    #lables_index = np.argmax(silhouette_coefficients)
    #cluster_labels = df["Cluster"]
    #print(cluster_labels)

    # Only Plotting below this line
    
    #X = TSNE(n_components=2, perplexity=5).fit_transform(features_pca_df)
    #tsne_df = pd.DataFrame()
    ##cluster_labels = pd.Series(labels[np.argmax(silhouette_coefficients)])
    #tsne_df["Image Name"] = df["Name"]
    #tsne_df['ClusterID'] = cluster_labels.values
    #tsne_df["X_tsne"]  = X[:, 0]
    #tsne_df["Y_tsne"] = X[:, 1]
    ##print(tsne_df)
    #fig, axes = plt.subplots(2, 1)
    #plt.style.use("fivethirtyeight")
    #axes[0].plot(range(min_clusters, max_clusters+1), sse)
    #axes[0].set_xlabel("Number of Clusters")
    #axes[0].set_ylabel("SSE")
    #axes[1].plot(range(min_clusters, max_clusters+1), silhouette_coefficients)
    #axes[1].set_xlabel("Number of Clusters")
    #axes[1].set_ylabel("Silhouette Coefficient")
    #plt.figure(figsize=(16,10))
    ##sns.scatterplot(
    ##    x="X_tsne", y="Y_tsne",
    ##    hue="ClusterID",
    ##    palette=sns.color_palette("hls", 10),
    ##    data=tsne_df,
    ##    legend="full",
    ##    alpha=1
    ##)
    #plt.show()



    #Image merging and showing/storing
