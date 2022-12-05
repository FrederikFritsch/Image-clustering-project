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
    
    # this version using the normalization method and PCA from the one in "KMeansCluster.py"
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
    # Clustering algorithm from file "clusteringAlgorithms.py"
    sse, score, silhouette_coefficients, labels = perform_HDBSCAN(features_pca_df, min_clusters, max_clusters)
    print(labels)
    results_df = image_names_df
    results_df["Cluster"] = pd.DataFrame(labels)
    #cluster_labels= pd.DataFrame(labels, columns=["Cluster"])
    #results_df = pd.concat([image_names_df, labels], axis=1)

    os.makedirs(f'{resultspath}', exist_ok=True)  
    results_df.to_csv(f'{resultspath}/HDBSCANResults.csv') 