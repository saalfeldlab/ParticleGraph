import scipy.cluster.hierarchy as hcluster
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt




class EmbeddingCluster:
    def __init__(self, model_config):
        self.model_config = model_config

    def get(self, data, method):
        if method == 'kmeans':
            kmeans = KMeans(init="random", n_clusters=self.model_config['ninteractions'], n_init=1000, max_iter=10000,
                            random_state=13)
            k = kmeans.fit(data)
            clusters= k.labels_
            nclusters = self.model_config['ninteractions']
        if method == 'distance':
            thresh = 1.5
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