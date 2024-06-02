import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class EmbeddingCluster:
    def __init__(self, config):
        self.n_interactions = config.simulation.n_interactions

    def get(self, data, method, thresh=2.5):
        match method:
            case 'kmeans':
                kmeans = KMeans(init="random", n_clusters=self.n_interactions, n_init=1000, max_iter=10000, random_state=10)
                k = kmeans.fit(data)
                clusters = k.labels_
                n_clusters = self.n_interactions
            case 'kmeans_auto':
                silhouette_avg_list = []
                silhouette_max = 0
                n_clusters = None
                for n in range(2, 10):
                    clusterer = KMeans(n_clusters=n, random_state=10, n_init='auto')
                    cluster_labels = clusterer.fit_predict(data)
                    silhouette_avg = silhouette_score(data, cluster_labels)
                    silhouette_avg_list.append(silhouette_avg)
                    if silhouette_avg > silhouette_max:
                        silhouette_max = silhouette_avg
                        n_clusters = n
                kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto')
                k = kmeans.fit(data)
                clusters = k.labels_

            case 'distance':
                clusters = hcluster.fclusterdata(data, thresh, criterion="distance") - 1
                n_clusters = len(np.unique(clusters))
            case 'distance':
                clusters = hcluster.fclusterdata(data, thresh, criterion="distance") - 1
                n_clusters = len(np.unique(clusters))
            case _:
                raise ValueError(f'Unknown method {method}')

        return clusters, n_clusters


if __name__ == '__main__':
    # generate 3 clusters of each around 100 points and one orphan point
    n_interactions = 3
    embedding_cluster = EmbeddingCluster(n_interactions)

    N = 100
    data = np.random.randn(3 * N, 2)
    data[:N] += 5
    data[-N:] += 10
    data[-1:] -= 20

    # clustering
    thresh = 1.5
    clusters, n_clusters = embedding_cluster.get(data, method="distance")

    # plotting
    plt.scatter(*np.transpose(data), c=clusters, s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, n_clusters)
    plt.title(title)
    plt.show()
