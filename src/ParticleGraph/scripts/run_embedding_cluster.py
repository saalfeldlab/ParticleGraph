# TODO: IN PROGRESS
from ParticleGraph.embedding_cluster import *
from GNN_particles_Ntype import cc
from ParticleGraph.utils import to_numpy

if __name__ == '__main__':

    model_config = {'ninteractions': 3, 'nparticles': 4800, 'nparticle_types': 3, 'cmap': 'tab10', 'model':'PDE_A'}

    cmap = cc(model_config=model_config)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    embedding_cluster = EmbeddingCluster(model_config)
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']

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

    proj_interaction = np.load('/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/log/try_arbitrary_3b/tmp_training/umap_projection_0.npy')

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1)
    plt.ion()

    labels, nclusters = embedding_cluster.get(proj_interaction,'distance')
    label_list = []

    for n in range(nparticle_types):
        tmp = labels[index_particles[n]]
        sub_group = np.round(np.median(tmp))
        label_list.append(sub_group)
    label_list = np.array(label_list)
    new_labels = labels.copy()
    for n in range(nparticle_types):
        new_labels[labels == label_list[n]] = n
        pos = np.argwhere(labels == label_list[n])
        plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1],
                    color=cmap.color(n), s=0.1)
    plt.xlabel(r'UMAP 0', fontsize=12)
    plt.ylabel(r'UMAP 1', fontsize=12)
    plt.text(0.05, 0.9, f'Nclusters: {nclusters}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    T1 = torch.zeros(int(nparticles / nparticle_types))
    for n in range(1, nparticle_types):
        T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types))), 0)
    T1 = T1[:, None]
    confusion_matrix = metrics.confusion_matrix(to_numpy(T1), new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
    plt.text(0.05, 0.8, f'Accuracy: {Accuracy}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(1, 2, 2)
    cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)


