import scipy.cluster.hierarchy as hcluster
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

class EmbeddingCluster:
    def __init__(self, model_config):
        self.model_config = model_config

    def get(self, data, method, thresh=2.5):
        if method == 'kmeans':
            kmeans = KMeans(init="random", n_clusters=self.model_config['ninteractions'], n_init=1000, max_iter=10000, random_state=10)
            k = kmeans.fit(data)
            clusters= k.labels_
            nclusters = self.model_config['ninteractions']
        if method == 'kmeans_auto':
            silhouette_avg_list = []
            silhouette_max = 0
            for n_clusters in range(2, 10):
                clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=1000)
                cluster_labels = clusterer.fit_predict(data)
                silhouette_avg = silhouette_score(data, cluster_labels)
                silhouette_avg_list.append(silhouette_avg)
                print(silhouette_avg)
                if silhouette_avg > silhouette_max:
                    silhouette_max = silhouette_avg
                    nclusters = n_clusters
            kmeans = KMeans(n_clusters=nclusters, random_state=10, n_init=1000)
            k = kmeans.fit(proj_interaction)
            clusters = k.labels_

        if method == 'distance':
            clusters = hcluster.fclusterdata(data, thresh, criterion="distance") - 1
            nclusters = len(np.unique(clusters))
        if method == 'distance':
            clusters = hcluster.fclusterdata(data, thresh, criterion="distance") - 1
            nclusters = len(np.unique(clusters))

        return clusters, nclusters


if __name__ == '__main__':
    # generate 3 clusters of each around 100 points and one orphan point

    model_config = {'ninteractions': 3}

    embedding_cluster = EmbeddingCluster(model_config)

    N = 100
    data = np.random.randn(3 * N, 2)
    data[:N] += 5
    data[-N:] += 10
    data[-1:] -= 20

    # clustering
    thresh = 1.5
    clusters, nclusters = embedding_cluster.get(data, method="distance")

    # plotting
    plt.scatter(*np.transpose(data), c=clusters,s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    plt.title(title)
    plt.show()

