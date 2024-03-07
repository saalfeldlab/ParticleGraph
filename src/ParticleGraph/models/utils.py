import torch
from prettytable import PrettyTable
from ParticleGraph.models import Interaction_Particles, Mesh_Laplacian, Mesh_RPS
from ParticleGraph.utils import choose_boundary_values
from ParticleGraph.utils import to_numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from ParticleGraph.fitting_models import linear_model
import umap

def get_embedding(model_a=None, dataset_number = 0, index_particles=None, n_particles=None, n_particle_types=None):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding

def plot_training (dataset_name, filename, log_dir, epoch, N, x, index_particles, n_particles, n_particle_types, model, dataset_num, ynorm, cmap, device):

    match filename:

        case 'embedding':
            fig = plt.figure(figsize=(8, 8))
            embedding = get_embedding(model.a, dataset_num, index_particles, n_particles, n_particle_types)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0],
                            embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/{filename}_{dataset_name}_{epoch}_{N}.tif", dpi=300)
            plt.close()
        case 'wave_mesh':
            rr = torch.tensor(np.linspace(-150, 150, 200)).to(device)
            popt_list = []
            for n in range(n_particles):
                embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
                h = model.lin_phi(in_features.float())
                h = h[:, 0]
                popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
                popt_list.append(popt)
            t = np.array(popt_list)
            t = t[:, 0]
            fig = plt.figure(figsize=(8, 8))
            embedding = get_embedding(model.a, 1)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=t[:, None], s=3, cmap='viridis')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/mesh_embedding_{dataset_name}_{epoch}_{N}.tif",dpi=300)
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            t = np.reshape(t, (100, 100))
            plt.imshow(t, cmap='viridis')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/mesh_map_{dataset_name}_{epoch}_{N}.tif",
                        dpi=300)
            # imageio.imwrite(f"./{log_dir}/tmp_training/embedding/{dataset_name}_map_{epoch}_{N}.tif", t, 'TIFF')

            fig = plt.figure(figsize=(8, 8))
            t = np.array(popt_list)
            t = t[:, 0]
            pts = x[:, 1:3].detach().cpu().numpy()
            tri = Delaunay(pts)
            colors = np.sum(t[tri.simplices], axis=1)
            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(), facecolors=colors)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/mesh_Delaunay_{dataset_name}_{epoch}_{N}.tif",
                        dpi=300)
            plt.close()

        case 'interaction':
            fig = plt.figure(figsize=(8, 8))
            rr = torch.tensor(np.linspace(0, radius, 200)).to(device)
            for n in range(n_particles):
                embedding_ = model.a[dataset_num, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                if (model_config.particle_model_name == 'PDE_A'):
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                elif (model_config.particle_model_name == 'PDE_A_bis'):
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                elif (model_config.particle_model_name == 'PDE_B'):
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                elif model_config.particle_model_name == 'PDE_E':
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                else:
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                func = model.lin_edge(in_features.float())
                func = func[:, 0]
                if n % 5 == 0:
                    plt.plot(to_numpy(rr),
                             to_numpy(func),
                             linewidth=1,
                             color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/{filename}_{dataset_name}_{epoch}_{N}.tif", dpi=300)
            plt.close()

def analyze_edge_function(rr=None, vizualize=False, config=None, model_lin_edge=[], model_a=None, dataset_number = 0, n_particles=None, ynorm=None, types=None, cmap=None, device=None):
    func_list = []
    for n in range(n_particles):
        embedding_ = model_a[dataset_number, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        max_radius = config.simulation.max_radius
        match config.graph_model.particle_model_name:
            case 'PDE_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_A_bis':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_, embedding_), dim=1)
            case 'PDE_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            case 'PDE_G':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, 0 * rr[:, None],
                                         0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            case 'PDE_E':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_, embedding_), dim=1)
        func = model_lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        if (n % 5 == 0) & vizualize:
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(types[n].astype(int)), linewidth=1, alpha=0.25)
    func_list = torch.stack(func_list)
    coeff_norm = to_numpy(func_list)
    if coeff_norm.shape[0] > 1000:
        new_index = np.random.permutation(coeff_norm.shape[0])
        new_index = new_index[0:min(1000, coeff_norm.shape[0])]
        trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0).fit(coeff_norm[new_index])
        proj_interaction = trans.transform(coeff_norm)
    else:
        trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
        proj_interaction = trans.transform(coeff_norm)
    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim([1E-3, 0.2])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)

    return func_list, proj_interaction

def choose_training_model(model_config, device):
    
    aggr_type = model_config.graph_model.aggr_type

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model=[]
    model_name = model_config.graph_model.particle_model_name
    match model_name:
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_bis' :
            model = Interaction_Particles(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
        case 'PDE_E':
            model = Interaction_Particles(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
        case 'PDE_G':
            model = Interaction_Particles(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
        case 'WaveMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
        case 'RD_RPS_Mesh':
            model = Mesh_RPS(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
  
    if model==[]:
        raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos

def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size

def set_trainable_parameters(model, lr_embedding, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    embedding = model.a
    optimizer = torch.optim.Adam([embedding], lr=lr_embedding)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def set_trainable_division_parameters(model, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.t)

    embedding = model.t
    optimizer = torch.optim.Adam([embedding], lr=lr)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params






