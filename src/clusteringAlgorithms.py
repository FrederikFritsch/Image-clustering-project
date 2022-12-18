from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def perform_KMeans(data, min_clusters, max_clusters):
    # --------- CALCULATE K-MEANS CLUSTERS ------------
    sse = []
    silhouette_coefficients = []
    labels = []
    cluster_distances = []
    for nr_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(init = "random", n_clusters = nr_clusters, n_init = 10, max_iter=300, random_state = 22)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)
        labels.append(kmeans.labels_)
        distance = kmeans.transform(data)**2
        cluster_distances.append(distance)
    return sse, silhouette_coefficients, labels, cluster_distances



def parameter_DBSCAN(data, min_epsilon, max_epsilon, min_samples, max_samples, resultspath):
    #print(f"input data:{data}")
    # --------- CALCULATE DBSCAN CLUSTERS ------------
    from itertools import product
    import pandas as pd
    import seaborn as sns
    epsilon_range = np.arange(min_epsilon, max_epsilon, 0.25) 
    min_samples_range = np.arange(min_samples, max_samples) 
    parameters = list(product(epsilon_range, min_samples_range))

    number_of_clusters = []
    silhouette_coefficients = []

    for i in parameters:
        clusterer = DBSCAN(eps=i[0], min_samples=i[1], metric='euclidean', metric_params=None, algorithm='auto', p=None, n_jobs=-1)
        clusterer = clusterer.fit(data)

        labels = clusterer.labels_
        number_of_clusters.append(len(np.unique(labels)))
        score = silhouette_score(data, labels)
        silhouette_coefficients.append(score)

    parameter_df = pd.DataFrame.from_records(parameters, columns =['Epsilon', 'Min_samples'])   
    parameter_df['silhouette_score'] = silhouette_coefficients
    # print(parameter_df)

    # for i in range(len(parameters)):
    ind = parameter_df['silhouette_score'].idxmax()
    row = parameter_df.iloc[ind,:]
    epsilon = row[0]
    min_sample = row[1].astype('int')
    print(min_sample)

    #    print(parameter_table[silhouette_coefficients][i])
    #    if parameter_table[silhouette_coefficients][i] == np.max(silhouette_coefficients):
    #        print(i)

    pivot_1 = pd.pivot_table(parameter_df, values='silhouette_score', index='Min_samples', columns='Epsilon')

    fig, ax = plt.subplots(figsize=(18,6))
    sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
    #plt.show()
    plt.savefig(f'{resultspath}/Parameters_Heatmap.png')

    #core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
    #core_samples_mask[clusterer.core_sample_indices_] = True
    #labels = clusterer.labels_
   
    return epsilon, min_sample



def perform_DBSCAN(data, epsilon, min_samples):
    #print(f"input data:{data}")
    # --------- CALCULATE DBSCAN CLUSTERS ------------
    silhouette_coefficients = []

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    db = db.fit(data)

    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    score = silhouette_score(data, labels)
    silhouette_coefficients.append(score)
   
    return labels, silhouette_coefficients



def parameter_HDBSCAN(data, min_cluster_size, max_cluster_size):
    #print(f"input data:{data}")
    import seaborn as sns
    import hdbscan

    relative_validities = []
    #labels = []
    #cluster_membership_scores = []
    print(f"input data shape:{data.shape}")

    cluster_size_range = []
    # the most important parameter is min_cluster_size
    for cluster_size in range(min_cluster_size, max_cluster_size+1):
        cluster_size_range.append(cluster_size)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, algorithm='best', alpha=1.0, approx_min_span_tree=True, 
        gen_min_span_tree=True, leaf_size=40, metric='euclidean', p=None, min_samples=1)

        clusterer.fit(data)
        #labels.append(clusterer.labels_)
        #score = silhouette_score(data, clusterer.labels_)
        relative_validities.append(clusterer.relative_validity_)
        #cluster_membership_scores.append(clusterer.probabilities_)

    index = np.argmax(relative_validities) 
    min_cluster_size = cluster_size_range[index]
    print(f"min cluster size:{min_cluster_size}")

    return min_cluster_size



def perform_HDBSCAN(data, min_cluster_size, resultspath):
    #print(f"input data:{data}")
    # --------- CALCULATE HDBSCAN CLUSTERS ------------
    import seaborn as sns
    import hdbscan

    relative_validities = []
    silhouette_coefficients = []
    labels = []
    print(f"input data shape:{data.shape}")

    # use t-SNE plot 
    projection = TSNE(init='pca', random_state=123, learning_rate='auto').fit_transform(data)
    plt.scatter(*projection.T,)
    #plt.show()
    plt.savefig(f'{resultspath}/TSNEPlotBeforeCluster.png')


    # the most important parameter is min_cluster_size
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, algorithm='best', alpha=1.0, approx_min_span_tree=True, 
                                gen_min_span_tree=True, leaf_size=40, metric='euclidean', min_samples=1, p=None)

    clusterer.fit(data)
    # print(clusterer.labels_)   # the labels of the clusters are [-1, -1, -1,..., -1] if the dataset only contain 6 images

    #Condensed Trees
    clusterer.condensed_tree_.plot(select_clusters=True)
    #plt.show()
    plt.savefig(f'{resultspath}/Condensed_Tree.png')

    labels = clusterer.labels_
    cluster_membership_score = clusterer.probabilities_
    relative_validities = clusterer.relative_validity_

    score = silhouette_score(data, labels)
    silhouette_coefficients.append(score)

    #After clustering with HDBSCAN and mapping the result to the t-SNE plot shown above.
    color_palette = sns.color_palette('deep', np.max(labels)+1)
    cluster_colors = [color_palette[i] if i >= 0 else (0.5, 0.5, 0.5) for i in labels]
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    #plt.show()
    plt.savefig(f'{resultspath}/TSNEPlotAfterCluster.png')

    return labels, cluster_membership_score, silhouette_coefficients, relative_validities