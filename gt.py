import graph_tool.all as gt
import matplotlib.cm
import numpy as np
import scipy.io


gt.seed_rng(47)

mat = scipy.io.loadmat("./graphs_data/Brain.mat")
adjacency = mat['A']
pos = np.argwhere(adjacency > 0)
w_ij = np.zeros(len(pos))

g = gt.Graph(directed=False)
g.add_vertex(len(pos))

index=0
for i in range(adjacency.shape[0]):
    for j in range(adjacency.shape[1]):
        if adjacency[i, j] >0:
            g.add_edge(i, j)
            w_ij[index] = adjacency[i, j]
            index += 1

weight = g.new_edge_property("double")
for index, e in enumerate(g.edges()):
    weight[e] = w_ij[index]

# g = gt.collection.ns["foodweb_baywet"]

sargs = dict(recs=[weight], rec_types=["real-exponential"])
state = \
    gt.minimize_nested_blockmodel_dl(g,
                                     state_args=sargs)

state.draw(edge_color=gt.prop_to_size(weight,
                                      power=1,
                                      log=True),
           ecmap=(matplotlib.cm.inferno, .6),
           eorder=weight,
           edge_pen_width=gt.prop_to_size(weight,
                                          1, 4,
                                          power=1,
                                          log=True),
           edge_gradient=[]);