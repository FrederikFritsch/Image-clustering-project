from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def perform_KMeans(data, min_clusters, max_clusters):
    # --------- CALCULATE K-MEANS CLUSTERS ------------
    sse = []
    silhouette_coefficients = []
    labels = []
    for nr_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(init = "random", n_clusters = nr_clusters, n_init = 10, max_iter=300, random_state = 22)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)     # HDBSCAN doesn't contain "inertia_"
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)
        labels.append(kmeans.labels_)
    return sse, score, silhouette_coefficients, labels


def perform_DBSCAN(data, min_cluster_size, max_cluster_size):
    import hdbscan
    print(data)
    # --------- CALCULATE DBSCAN CLUSTERS ------------

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    #sse = []
    silhouette_coefficients = []
    #labels = []
    #not allow unclustered points
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=5, metric='euclidean', min_cluster_size=5, min_samples=None, p=None)#cluster_selection_epsilon = 0.00001)
    clusterer.fit(data)
    print(f"Cluster Lables: {clusterer.labels_}")   # the labels of the clusters are [-1, -1, -1, .... -1]
    
    #sse.append(0)       # DBSCAN doesn't contain "inertia_"
    score = silhouette_score(data, clusterer.labels_)
    silhouette_coefficients.append(score)
    labels.append(clusterer.labels_)    
    return score, silhouette_coefficients, labels


def perform_HDBSCAN(data, min_cluster_size, max_cluster_size):
    import hdbscan
    print(data)
    # --------- CALCULATE HDBSCAN CLUSTERS ------------
    #sse = []
    silhouette_coefficients = []
    labels = []
    #not allow unclustered points
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon = 0.0001, min_cluster_size = min_cluster_size, min_samples = 1)
    clusterer.fit(data)
    print(clusterer.labels_)   # the labels of the clusters are [-1, -1, -1, .... -1]
    
    #sse.append(0)       # HDBSCAN doesn't contain "inertia_"
    score = silhouette_score(data, clusterer.labels_)
    #sse.append(0)
    score = 0#silhouette_score(data, clusterer.labels_)
    silhouette_coefficients.append(score)
    labels = clusterer.labels_

    
    return score, silhouette_coefficients, labels
