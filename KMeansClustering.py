import numpy as np
import pandas as pd
import os
import time
import sys
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.decomposition import PCA
from src.clusteringAlgorithms import *
from src.featureExtraction import *
from src.Utils import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        filepath = str(args[0])
        resultspath = str(args[1])
        normalization_method = str(args[2])
        pca_variance = float(args[3])
        min_clusters = int(args[4])
        max_clusters = int(args[5])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()


    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(e)
        quit()
    #print(df)
    # Feature normalization
    features_df = df.loc[:, df.columns[1:]]
    image_names_df = df.loc[:, ['Name']]
    

    if normalization_method == "normalize":
        print("Normalizing")
        scaler = MinMaxScaler()
        features_df = scaler.fit_transform(features_df)
    else:
        scaler = StandardScaler()
        print("Standardizing")
        features_df = scaler.fit_transform(features_df)
    #print(features_df)
    # Dimensionality reduction
    pca = PCA(pca_variance)
    
    features_pca_df = pca.fit_transform(features_df)

    print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering
    sse, silhouette_coefficients, labels, cluster_distances= perform_KMeans(features_pca_df, min_clusters, max_clusters)
    
    #print(sse)
    #print(silhouette_coefficients)

    # Cluster number evaluation

    n_clusters = np.argmax(silhouette_coefficients) + min_clusters
    print(f"Silhouette coefficient: {n_clusters} clusters return best results")
    
    index = np.argmax(silhouette_coefficients)
    #print(len(labels[index]))
    
    X_dist = cluster_distances[index]
    center_dists = np.array([round(X_dist[i][x], 2) for i,x in enumerate(labels[index])])
    #print(center_dists)
    cluster_labels = pd.DataFrame(labels[index], columns=["Cluster"])
    cluster_distance = pd.DataFrame(center_dists, columns=["Distance"])
    results_df = pd.concat([image_names_df, cluster_labels, cluster_distance], axis=1)
    print(results_df)

    os.makedirs(f'{resultspath}', exist_ok=True)  
    results_df.to_csv(f'{resultspath}/KMeansResults.csv') 

    fig, axes = plt.subplots(2, 1)
    plt.style.use("fivethirtyeight")
    axes[0].plot(range(min_clusters, max_clusters+1), sse)
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("SSE")
    axes[1].plot(range(min_clusters, max_clusters+1), silhouette_coefficients)
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Coefficient")

    plt.savefig(f'{resultspath}/ClusterScores.png')
    #plt.show()
    