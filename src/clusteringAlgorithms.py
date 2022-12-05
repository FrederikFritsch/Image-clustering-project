from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan

def perform_KMeans(data, min_clusters, max_clusters):
    # --------- CALCULATE K-MEANS CLUSTERS ------------
    sse = []
    silhouette_coefficients = []
    labels = []
    for nr_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(init = "random", n_clusters = nr_clusters, n_init = 10, max_iter=300, random_state = 22)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)
        labels.append(kmeans.labels_)
    return sse, score, silhouette_coefficients, labels


def perform_HDBSCAN(data, min_cluster_size, max_cluster_size):
    print(data)
    # --------- CALCULATE HDBSCAN CLUSTERS ------------
    sse = []
    silhouette_coefficients = []
    labels = []
    #not allow unclustered points
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon = 0.00001)
    clusterer.fit(data)
    print(clusterer.labels_)   # the labels of the clusters are [-1, -1, -1, .... -1]
    
    sse.append(0)
    score = silhouette_score(data, clusterer.labels_)
    silhouette_coefficients.append(score)
    labels.append(clusterer.labels_)

    
    return sse, score, silhouette_coefficients, labels