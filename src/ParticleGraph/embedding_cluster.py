import scipy.cluster.hierarchy as hcluster
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from GNN_particles_Ntype import *




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

    plotting
    plt.scatter(*np.transpose(data), c=clusters,s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    plt.title(title)
    plt.show()

    config = '/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/config/config_arbitrary_3.yaml'

    # Load parameters from config file
    with open(f'{config}', 'r') as file:
        model_config = yaml.safe_load(file)
    model_config['dataset']=config[7:]
    embedding_cluster = EmbeddingCluster(model_config)


    for key, value in model_config.items():
        print(key, ":", value)
        if ('E-' in str(value)) | ('E+' in str(value)):
            value = float(value)
            model_config[key] = value

    cmap = cc(model_config=model_config)
    aggr_type = model_config['aggr_type']

    if model_config['boundary'] == 'no':  # change this for usual BC
        def bc_pos(X):
            return X


        def bc_diff(D):
            return D
    else:
        def bc_pos(X):
            return torch.remainder(X, 1.0)


        def bc_diff(D):
            return torch.remainder(D - .5, 1.0) - .5

    model = []
    radius = model_config['radius']
    min_radius = model_config['min_radius']
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']
    dataset_name = model_config['dataset']
    nframes = model_config['nframes']
    bMesh = (model_config['model'] == 'DiffMesh') | (model_config['model'] == 'WaveMesh')
    nrun = model_config['nrun']
    kmeans_input = model_config['kmeans_input']
    aggr_type = model_config['aggr_type']

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(dataset_name))
    print('log_dir: {}'.format(log_dir))

    model = InteractionParticles(model_config=model_config, device=device, aggr_type = model_config['aggr_type'], bc_diff=bc_diff)
    print(f'Cluster InteractionParticles')

    net = f"/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/log/try_arbitrary_3/models/best_model_with_1_graphs_20.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    fig = plt.figure(figsize=(8, 8))



    ax = fig.add_subplot(1, 4, 1)
    for n in range(nparticle_types):
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=5, alpha=0.75)
    plt.xlabel('UMAP 0', fontsize=12)
    plt.ylabel('UMAP 1', fontsize=12)

    kmeans = KMeans(init="random", n_clusters=model_config['ninteractions'], n_init=5000, max_iter=10000,
                    random_state=13)
    if kmeans_input == 'plot':
        kmeans.fit(proj_interaction)
    if kmeans_input == 'embedding':
        kmeans.fit(embedding_)
    for n in range(nparticle_types):
        tmp = kmeans.labels_[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        accuracy = len(np.argwhere(tmp == sub_group)) / len(tmp) * 100
        print(f'Sub-group {n} accuracy: {np.round(accuracy, 3)}')
    for n in range(model_config['ninteractions']):
        plt.plot(kmeans.cluster_centers_[n, 0], kmeans.cluster_centers_[n, 1], '+', color='k', markersize=12)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/Fig_{dataset_name}_{epoch}.tif")