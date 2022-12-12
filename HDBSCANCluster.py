import numpy as np
import pandas as pd
import os
import time
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.preprocessing import normalize
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
        pca_variance = float(args[3])     # how to decide the value of this by ourselves?
        #min_clusters = int(args[4])       # try several values of min_clusters and max_clusters, the results are the same [-1, -1,...,-1]
        #max_clusters = int(args[5])
        min_cluster_size = int(args[4])
        #max_cluster_size = int(args[5])
    except Exception as e:
        print("Wrong usage of arguments.")
        print(e)
        quit()
    

    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(e)
        quit()

    # print(df) # print dataframe with features and image names with colomn names # the shape is (208, 1192)

    # separate features and image names
    features_df = df.loc[:, df.columns[1:]]  # get all features without column names
    image_names_df = df.loc[:, ['Name']]     # get all image names with column name
    
    if normalization_method == "normalize":      # K-means use "Normalize"
        print("Normalizing")
        scaler = MinMaxScaler()
        features_df = scaler.fit_transform(features_df)
    else:
        scaler = StandardScaler()
        print("Standardizing")
        #scaler.fit(features_df)
        features_df = scaler.fit_transform(features_df)  # DBSCAN and HDBSCAN use "Standardize" # equal to "StandardScaler().fit_transform(features_df)"

    #print(features_df) # features after normalization/standardization # the shape is (208, 1191)

    # Dimensionality reduction
    #pca = PCA(n_components=2)
    pca = PCA(pca_variance)       
    features_pca_df = pca.fit_transform(features_df)   # the shape is (208, 32)

    a = np.array(features_pca_df[0]).T.tolist()
    b = np.array(features_pca_df[1]).T.tolist()
    df2 = pd.DataFrame({'Feature_PCA': [a]})
    df2.loc[len(df2.index)]=[b]
    print(df2)

    ######## have a try ##########
    '''
    def scatter_thumbnails(data, images, zoom=0.12, colors=None):
        assert len(data) == len(images)

        x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

        # create a scatter plot.
        f = plt.figure(figsize=(22, 15))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], s=4)
        #sc = ax.scatter(features_pca_df[:,0], features_pca_df[:,1], s=4)
        _ = ax.axis('off')
        _ = ax.axis('tight')

        # add thumbnails
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        for i in range(len(images)):
            image = plt.imread(images[i])
            im = OffsetImage(image, zoom=zoom)
            bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
            ab = AnnotationBbox(im, x[i], xycoords='features_df',
                                frameon=(bboxprops is not None),
                                pad=0.02,
                                bboxprops=bboxprops)
            ax.add_artist(ab)
        return ax

    scatter_thumbnails(features_df.tolist(), df.Name.tolist())
    plt.title('Image Visualization - Principal Component Analysis')
    plt.show()
    '''
    ######## until here #########

   
    print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering algorithm from file "clusteringAlgorithms.py"
    labels, cluster_membership_score, silhouette_coefficients = perform_HDBSCAN(features_pca_df, min_cluster_size)
    print(f"labels of HDBSCAN:{labels}")

    results_df = image_names_df
    results_df["Cluster"] = pd.DataFrame(labels)
    results_df["Cluster_membership_score"] = pd.DataFrame(cluster_membership_score)
    print(f"Silhouette Score of HDBSCAN is :{silhouette_coefficients}")

    
    #print(results_df)

    #cluster_labels= pd.DataFrame(labels, columns=["Cluster"])
    #results_df = pd.concat([image_names_df, labels], axis=1)

    os.makedirs(f'{resultspath}', exist_ok=True)  
    results_df.to_csv(f'{resultspath}/HDBSCANResults.csv') 