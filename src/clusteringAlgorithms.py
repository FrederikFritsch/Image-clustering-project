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



def perform_DBSCAN(data, min_cluster, max_cluster):  # didn't finish, but this is similar with HDBSCAN
    #print(f"input data:{data}")
    # --------- CALCULATE DBSCAN CLUSTERS ------------

    silhouette_coefficients = []
    #labels = []

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    #sse.append(0)       
    # DBSCAN doesn't contain "inertia_"
    score = 0# silhouette_score(data, clusterer.labels_)
    silhouette_coefficients.append(score)
    #labels.append(clusterer.labels_)    
    return score, silhouette_coefficients, labels


def perform_HDBSCAN(data, min_cluster_size, max_cluster_size, resultspath):
    import seaborn as sns
    import hdbscan
    print(f"input data:{data}")
    # --------- CALCULATE HDBSCAN CLUSTERS ------------
    relative_validities = []
    labels = []
    cluster_membership_scores = []
    print(f"input data shape:{data.shape}")

    # use t-SNE plot 
    projection = TSNE(init='pca', random_state=123, learning_rate='auto').fit_transform(data)
    plt.scatter(*projection.T,)
    plt.savefig(f'{resultspath}/TSNEPlotBeforeCluster.png')

    # the most important parameter is min_cluster_size
    for cluster_size in range(min_cluster_size, max_cluster_size+1):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, algorithm='best', approx_min_span_tree=True, 
        gen_min_span_tree=True, leaf_size=40, metric='euclidean', p=None)
        clusterer.fit(data)
        labels.append(clusterer.labels_)
        #score = silhouette_score(data, clusterer.labels_)
        relative_validities.append(clusterer.relative_validity_)
        cluster_membership_scores.append(clusterer.probabilities_)
    #print(clusterer.labels_)   # the labels of the clusters are [-1, -1, -1,..., -1] if the dataset only contain 6 images
    
    #Condensed Trees
    clusterer.condensed_tree_.plot(select_clusters=True)
    plt.savefig(f'{resultspath}/Condensed_Tree.png')
    
    try:
        # After clustering with HDBSCAN and mapping the result to the t-SNE plot shown above.
        color_palette = sns.color_palette()
        #cluster_colors = [color_palette[x] if x >= 0
        #                else (0.5, 0.5, 0.5)
        #                for x in labels]
        #plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
        #plt.show()
    except:
        pass
    return labels, cluster_membership_scores, relative_validities
