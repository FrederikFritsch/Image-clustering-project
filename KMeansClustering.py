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
        #scaler.fit(features_df)
        features_df = scaler.fit_transform(features_df)
    print(features_df)
    # Dimensionality reduction
    pca = PCA(pca_variance)
    
    features_pca_df = pca.fit_transform(features_df)

    print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering
    sse, score, silhouette_coefficients, labels = perform_KMeans(features_pca_df, min_clusters, max_clusters)
    

    # Cluster number evaluation

    n_clusters = np.argmax(silhouette_coefficients) + min_clusters
    print(f"Silhouette coefficient: {n_clusters} clusters return best results")
    
    lables_index = np.argmax(silhouette_coefficients)
    cluster_labels = pd.DataFrame(labels[lables_index], columns=["Cluster"])

    results_df = pd.concat([image_names_df, cluster_labels], axis=1)
    #print(results_df)

    # Only Plotting below this line

    #fig, axes = plt.subplots(2, 1)
    #plt.style.use("fivethirtyeight")
    #axes[0].plot(range(min_clusters, max_clusters+1), sse)
    #axes[0].set_xlabel("Number of Clusters")
    #axes[0].set_ylabel("SSE")
    #axes[1].plot(range(min_clusters, max_clusters+1), silhouette_coefficients)
    #axes[1].set_xlabel("Number of Clusters")
    #axes[1].set_ylabel("Silhouette Coefficient")
    #plt.figure(figsize=(16,10))

    os.makedirs(f'{resultspath}', exist_ok=True)  
    results_df.to_csv(f'{resultspath}/ClusterResults.csv') 