import argparse
import torch
import numpy as np
from motile_toolbox.candidate_graph.graph_attributes import NodeAttr, EdgeAttr
from motile.track_graph import TrackGraph
from motile.solver import Solver
from motile.constraints import MaxChildren, MaxParents 
from motile.costs import EdgeSelection, EdgeDistance
from motile.variables import EdgeSelected, NodeSelected
from motile.plot import draw_track_graph, draw_solution
import matplotlib
from matplotlib import pyplot as plt
import tifffile
from motile_toolbox.candidate_graph.compute_graph import get_candidate_graph_from_points_list
from scipy import ndimage

def track(points_file_name):
    
    # read the pt file
    pts = torch.load(points_file_name, map_location=torch.device('cpu'))
    n_frames = len(pts)

    for i in range(len(pts)):
        pts_t = pts[i]
        pts_temp = pts_t[:, :3]
        pts_temp[:, 0] = i
        if i==0:
            points_numpy =pts_temp
        else:
            points_numpy = np.concatenate((points_numpy, pts_temp), axis=0)
    
    print(f"Loaded data has shape {points_numpy.shape}")

    
    # t [z] y x  

    # make a candidate graph on the points
    candidate_graph = get_candidate_graph_from_points_list(points_numpy, max_edge_distance =0.1) # TODO    
    
    # convert this to a motile track graph
    track_graph = TrackGraph(candidate_graph, frame_attribute="time")
    print(f"Number of nodes is {len(track_graph.nodes)}, number of edges is {len(track_graph.edges)}")


    # initialize the solver
    solver = Solver(track_graph = track_graph)        

    # specify constraints
    # two children
    # one parent
    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(2))

    # specify the costs
    # here will go over some features
    solver.add_costs(EdgeDistance(weight=1.0, position_attribute=NodeAttr.POS.value, constant=-0.05), name= "Distance")
    # add more costs ....

    # solve
    print(f"Solver ...")
    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)
    
    
    # look at the solution
    nodes = solver.get_variables(NodeSelected)   
    edges = solver.get_variables(EdgeSelected)   


    selected_nodes = [node for node in track_graph.nodes if solution[nodes[node]] > 0.5]
    selected_edges = [edge for edge in track_graph.edges if solution[edges[edge]] > 0.5]

    print(f"length of selected nodes is {len(selected_nodes)}")
    print(f"length of selected edges is {len(selected_edges)}")

    # initialize trajectory: there are len(pts[0]) at first
    trajectory = {}
    for k in range(len(pts[0])):
        trajectory[k] = [k]
    trajectory_id = len(pts[0])

    for edges_ in selected_edges:
        flag=True
        t = 0
        while(flag):
            if edges_[0]==trajectory[t][-1]:
                trajectory[t].append(int(edges_[1]))
                flag = False
            else:
                t+=1
            if t==trajectory_id:
                flag = False
        if t == trajectory_id:
            trajectory[trajectory_id] = [edges_[0], edges_[1]]
            trajectory_id+=1

    for k in range(trajectory_id-1,-1,-1):
        trajectory[k+1]=trajectory[k]
    trajectory[0]=[0]



    for t in range(len(pts)):
        point_list = []

        for k in range(1, len(trajectory)):
            pos = np.argwhere(points_numpy[trajectory[k],0]==t)
            if len(pos)>0:
                pos=pos[0][0]
                p_ = points_numpy[trajectory[k][pos]].copy()
                p_[0] = k
                point_list.append(p_.squeeze())
        point_list=np.array(point_list)
        fig = plt.figure(figsize=(12, 12))
        plt.ion()
        plt.scatter (point_list[:,1], point_list[:,2], s=10, c='b')
        for k in range(len(point_list)):
            plt.text(point_list[k][1], point_list[k][2], str(int(point_list[k][0])), fontsize=8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        num = f"{t:06}"
        plt.savefig(f"tmp/fig/trajectory_{num}.png")
        plt.close()

        size = 1024
        radius = 4

        image = np.zeros((size, size))
        for i in range(len(point_list)):
            point_i = (point_list[i,2]*1024).astype(int)
            point_j = (point_list[i,1]*1024).astype(int)
            y, x = np.ogrid[-point_i: size - point_i, -point_j: size - point_j]
            mask = x * x + y * y <= radius * radius
            image[mask] = point_list[i,0].astype(int)
        image = np.flipud(image)
        image = np.uint16(image)
        tifffile.imwrite(f"tmp/mask/trajectory_{num}.tif", image, photometric='minisblack')






        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_file_name", dest="points_file_name", default="./x_list_0.pt")
    args = parser.parse_args()
    track(args.points_file_name)


