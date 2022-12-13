import numpy as np
import pandas as pd
import os
import time
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
# from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from src.clusteringAlgorithms import *
from src.featureExtraction import *
from src.Utils import *
import seaborn as sns



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Missing arguments")
    try:
        filepath = str(args[0])
        filepath = "Results/" + filepath
        resultspath = str(args[1])
        resultspath = "Results/" + resultspath
        normalization_method = str(args[2])
        pca_variance = float(args[3])     # how to decide the value of this by ourselves?
        #min_clusters = int(args[4])      
        #max_clusters = int(args[5])
        min_cluster_size = int(args[4])   # the most important parameter for HDBSCAN is min_cluster_size
        max_cluster_size = int(args[5])   # this is only used in the for loop
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
    # get all features without column names
    features_df = df.loc[:, df.columns[1:]]
    # get all image names with column name
    image_names_df = df.loc[:, ['Name']]

    if normalization_method == "MinMax":
        print("Min Max transforming")
        scaler = MinMaxScaler()
        features_df = scaler.fit_transform(features_df)
    elif normalization_method == "Normalize":
        print("Normalizing")
        scaler = Normalizer()
        features_df = scaler.fit_transform(features_df)
    else:
        scaler = StandardScaler()
        print("Standardizing")
        # DBSCAN and HDBSCAN use "Standardize" # equal to "StandardScaler().fit_transform(features_df)"
        features_df = scaler.fit_transform(features_df)

    # print(features_df) # features after normalization/standardization # the shape is (208, 1191)

    # Dimensionality reduction
    pca = PCA(pca_variance)
    features_pca_df = pca.fit_transform(features_df)   # the shape is (208, 32)

    print(f"The dimensions of features after PCA:{features_pca_df.shape}")


    # reformat the features after PCA to plot the images
    a = np.array(features_pca_df[0]).T.tolist()
    b = np.array(features_pca_df[1]).T.tolist()
    df2 = pd.DataFrame({'Feature_PCA': [a]})
    df2.loc[len(df2.index)] = [b]
    for i in range(2, 208):
        c = np.array(features_pca_df[i]).T.tolist()
        df2.loc[len(df2.index)]=[c]


    def scatter_thumbnails(data, images, zoom=0.12, colors=None):
        assert len(data) == len(images)

        x = PCA(n_components=2).fit_transform(data) if len(
            data[0]) > 2 else data  # do PCA again to get 2 components

        # create a scatter plot.
        f = plt.figure(figsize=(22, 15))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], s=4)
        _ = ax.axis('off')
        _ = ax.axis('tight')

        # add thumbnails
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        for i in range(len(images)):
            image = plt.imread(images[i])
            im = OffsetImage(image, zoom=zoom)
            bboxprops = dict(
                edgecolor=colors[i]) if colors is not None else None
            ab = AnnotationBbox(im, x[i], xycoords='data',
                                frameon=(bboxprops is not None),
                                pad=0.02,
                                bboxprops=bboxprops)
            ax.add_artist(ab)
        return ax

    scatter_thumbnails(df2.Feature_PCA.tolist(), df.Name.tolist())
    plt.title('Image Visualization after PCA')
    #plt.show()
    plt.savefig(f'{resultspath}/VisualizationPCA.png')

    # use t-SNE to visualize the images, you can skip this part if you want
    from sklearn.manifold import TSNE
    x1 = PCA().fit_transform(df2['Feature_PCA'].tolist())
    x1 = TSNE(perplexity=50, n_components=2, init='pca', random_state=123,
              learning_rate='auto').fit_transform(x1)  # you can also try n_components = 3
    _ = scatter_thumbnails(x1, df.Name.tolist(), zoom=0.06)
    plt.title('2D t-Distributed Stochastic Neighbor Embedding')
    #plt.title('3D t-Distributed Stochastic Neighbor Embedding')
    plt.savefig(f'{resultspath}/2D-TSNE.png')
    #plt.show()

    print(f"Explained components: {pca.explained_variance_ratio_}")

    # Clustering algorithm from file "clusteringAlgorithms.py"
    min_cluster_size = parameter_HDBSCAN(features_pca_df, min_cluster_size, max_cluster_size)
    labels, cluster_membership_score, silhouette_coefficients, relative_validities = perform_HDBSCAN(features_pca_df, min_cluster_size, resultspath)
    
    print(f"labels of HDBSCAN:{labels}")

    palette = sns.color_palette('deep', np.max(labels) + 1)
    colors = [palette[i] if i >= 0 else (0, 0, 0) for i in labels]
    ax = scatter_thumbnails(features_pca_df, df.Name.tolist(), 0.06, colors)
    plt.title(f'Clusters by using HDBSCAN')
    #plt.show()
    plt.savefig(f'{resultspath}/HDBSCANClusters.png')

    # Number of clusters in labels, ignoring noise if present.
    HDBSCAN_number_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    results_df = image_names_df
    results_df["Cluster"] = pd.DataFrame(labels)
    results_df["Cluster_membership_score"] = pd.DataFrame(cluster_membership_score)
    print(f"Relative Validity of HDBSCAN is :{relative_validities}")
    print(f"The number of clusters of HDBSCAN: {HDBSCAN_number_clusters}")


    os.makedirs(f'{resultspath}', exist_ok=True)
    results_df.to_csv(f'{resultspath}/HDBSCANResults.csv')
