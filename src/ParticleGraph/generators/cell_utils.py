import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d,  Delaunay
import torch
from ParticleGraph.utils import to_numpy
import math
import torch_geometric.data as data
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tifffile import imread, imsave
import glob
from skimage.measure import regionprops
import os, time, argparse
import numpy as np
import pandas as pd
from skimage import io
import vedo
from joblib import Parallel, delayed

def init_cell_range(config, device, scenario="None"):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types

    ##### defines all variables for the cell model, per type of cell: dimension = n_particle_types

    if config.simulation.cell_cycle_length != [-1]:
        cycle_length = torch.tensor(config.simulation.cell_cycle_length, device=device)
    else:
        cycle_length = torch.clamp(torch.abs(torch.ones(n_particle_types, 1, device=device) * 250 + torch.randn(n_particle_types, 1, device=device) * 50), min=100, max=700).squeeze()

    if config.simulation.final_cell_mass != [-1]:
        final_cell_mass = torch.tensor(config.simulation.final_cell_mass, device=device)
    else:
        final_cell_mass = torch.clamp(torch.abs(
            torch.ones(n_particle_types, 1, device=device) * 250 + torch.randn(n_particle_types, 1,
                                                                               device=device) * 25), min=200,
                                      max=500).flatten()

    if config.simulation.cell_death_rate != [-1]:
        cell_death_rate = torch.tensor(config.simulation.cell_death_rate, device=device)
    else:
        cell_death_rate = torch.zeros((n_particles, 1), device=device)

    if config.simulation.cell_area != [-1]:
        cell_area = torch.tensor(config.simulation.cell_area, device=device)
    else:
        cell_area = torch.clamp(torch.abs(torch.ones(n_particle_types, 1, device=device) * 0.0015 + torch.randn(n_particle_types, 1, device=device) * 0.0010), min=0.0005, max=0.0025).squeeze()

    return cycle_length, final_cell_mass, cell_death_rate, cell_area

def init_cells(config, cycle_length, final_cell_mass, cell_death_rate, cell_area, bc_pos, bc_dpos, dimension, device):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if (simulation_config.boundary == 'periodic'):  # | (simulation_config.dimension == 3):

        pos = torch.rand(1, dimension, device=device)
        count = 1
        intermediate_count = 0
        distance_threshold = 0.025
        while count < n_particles:
            new_pos = torch.rand(1, dimension, device=device)
            distance = torch.sum(bc_dpos(pos[:, None, :] - new_pos[None, :, :]) ** 2, dim=2)
            if torch.all(distance > distance_threshold**2):
                pos = torch.cat((pos, new_pos), 0)
                count += 1
            intermediate_count += 1
            if intermediate_count > 100:
                distance_threshold = distance_threshold * 0.99
                intermediate_count = 0

    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5

    ###### specify all variables per cell, dimension = n_particles

    # specify position
    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    # specify type
    if config.simulation.cell_type_map is not None:
        i0 = imread(f'graphs_data/{config.simulation.cell_type_map}')
        type_values = np.unique(i0)
        i0_ = np.zeros_like(i0)
        for n, pixel_values in enumerate(type_values):
            i0_[i0 == pixel_values] = n
        type = i0_[255-(to_numpy(pos[:, 1]) * 255).astype(int), (to_numpy(pos[:, 0]) * 255).astype(int)].astype(int)
        type = torch.tensor(type, device=device)
        type = torch.clamp(type, min=0, max=n_particle_types - 1)
    else:
        type = torch.zeros(int(n_particles / n_particle_types), device=device)
        for n in range(1, n_particle_types):
            type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        if (simulation_config.params == 'continuous') | (
                config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
            type = torch.tensor(np.arange(n_particles), device=device)
    # specify cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
    status = torch.ones(n_particles, 2, device=device)
    status[:, 1] = 0

    cycle_length_distrib = cycle_length[to_numpy(type)] * (
                torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))
    cycle_length_distrib = cycle_length_distrib[:, None]

    cell_age = torch.rand(n_particles, device=device)
    cell_age = cell_age * cycle_length[to_numpy(type)].squeeze()
    cell_age = cell_age[:, None]
    cell_stage = update_cell_cycle_stage(cell_age, cycle_length, type, device)

    growth_rate = final_cell_mass / (2 * cycle_length)
    growth_rate_distrib = growth_rate[to_numpy(type)].squeeze()[:, None]

    cell_mass_distrib = (growth_rate_distrib * cell_age) + (final_cell_mass[to_numpy(type), None] / 2)

    cell_death_rate_distrib = (cell_death_rate[to_numpy(type)].squeeze() * (
                torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))) / 100
    cell_death_rate_distrib = cell_death_rate_distrib[:, None]

    cell_area_distrib = cell_area[to_numpy(type)].squeeze()[:, None]

    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    type = type[:, None]

    perimeter = torch.zeros((n_particles,1), device=device)

    return particle_id, pos, dpos, type, status, cell_age, cell_stage, cell_mass_distrib, growth_rate_distrib, cycle_length_distrib, cell_death_rate_distrib, cell_area_distrib, perimeter

def update_cell_cycle_stage(cell_age, cycle_length, type_list, device):
    g1 = 0.46
    s = 0.33
    g2 = 0.17
    m = 0.04

    G1 = (g1 * cycle_length).squeeze()
    S = ((g1 + s) * cycle_length).squeeze()
    G2 = ((g1 + s + g2) * cycle_length).squeeze()
    M = ((g1 + s + g2 + m) * cycle_length).squeeze()

    cell_age = cell_age.squeeze()

    cell_stage = torch.zeros(len(cell_age), device=device)
    for i in range(len(cell_age)):
        curr = cell_age[i]

        if curr <= G1[int(type_list[i])]:
            cell_stage[i] = 0
        elif curr <= S[int(type_list[i])]:
            cell_stage[i] = 1
        elif curr <= G2[int(type_list[i])]:
            cell_stage[i] = 2
        else:
            cell_stage[i] = 3

    return cell_stage[:, None]

def get_vertices(points=[], device=[]):

    all_points = points
    if points.shape[1] == 3:   # has 3D
        v_list = [[-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, -1, 1], [0, -1, 1], [0, 0, 1],
                  [-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, -1, 0], [0, -1, 0],
                  [-1, -1, -1], [-1, 0, -1], [-1, 1, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1], [1, -1, -1], [0, -1, -1], [0, 0, -1]]
    else:
        v_list = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    v_list = torch.tensor(v_list, device=device)
    for n in range(len(v_list)):
        all_points = torch.concatenate((all_points, points + v_list[n]), axis=0)

    if points.shape[1] == 3:
        pos = torch.argwhere((all_points[:, 0] > -0.05) & (all_points[:, 0] < 1.05) & (all_points[:, 1] > -0.05) & (
                    all_points[:, 1] < 1.05) & (all_points[:, 2] > -0.05) & (all_points[:, 2] < 1.05))
    else:
        pos = torch.argwhere ((all_points[:,0] >-0.05) & (all_points[:,0] <1.05) & (all_points[:,1] >-0.05) & (all_points[:,1] <1.05))
    all_points = all_points[pos].squeeze()

    vor = Voronoi(to_numpy(all_points))

    # fig = plt.figure(figsize=(10, 10))
    # voronoi_plot_2d(vor, ax=fig.gca(), show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5)
    # plt.scatter(to_numpy(points[:, 0]), to_numpy(points[:, 1]), s=30, color='red')

    # vertices_index collect all vertices index of regions of interest
    vertices_per_cell = []
    for n in range(len(points)):
        if n == 0:
            vertices_index = vor.regions[vor.point_region[0].copy()]
        else:
            vertices_index = np.concatenate((vertices_index, vor.regions[vor.point_region[n]]), axis=0)
        vertices_per_cell.append((vor.regions[vor.point_region[n]].copy()))

    vertices = []
    map = {}
    count = 0
    for i in range(len(vertices_per_cell)):
        for j in range(len(vertices_per_cell[i])):
            if vertices_per_cell[i][j] in map:
                vertices_per_cell[i][j] = map[vertices_per_cell[i][j]]
            else:
                map[vertices_per_cell[i][j]] = count
                vertices.append(vor.vertices[vertices_per_cell[i][j]])
                vertices_per_cell[i][j] = map[vertices_per_cell[i][j]]
                count += 1
    vertices_pos = np.array(vertices)
    vertices_pos = torch.tensor(vertices_pos, device=device)
    vertices_pos = vertices_pos.to(dtype=torch.float32)

    return vor, vertices_pos, vertices_per_cell, all_points

def get_Delaunay(points=[], device=[]):

    tri = Delaunay(to_numpy(points))  # Compute Delaunay triangulation

    p = points[tri.simplices]  # Triangle vertices

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T

    # fig = plt.figure()
    # plt.scatter(to_numpy(A[0, 100]), to_numpy(A[1,100]), s=10, color='blue')
    # plt.scatter(to_numpy(B[0, 100]), to_numpy(B[1,100]), s=10, color='blue')
    # plt.scatter(to_numpy(C[0, 100]), to_numpy(C[1,100]), s=10, color='blue')

    # Compute circumcenters (cc)
    a = A - C
    b = B - C

    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2 * ncross2(a, b) + 1E-16) + C

    # plt.scatter(to_numpy(cc[0, 100]), to_numpy(cc[1,100]), s=10, color='red')

    cc = cc.t()

    return cc, tri.simplices

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)

def get_voronoi_areas(vertices_pos, vertices_per_cell, device):

    centroids = get_voronoi_centroids(vertices_pos, vertices_per_cell, device)
    areas = []
    
    for i in range(len(vertices_per_cell)):
        v_list = vertices_per_cell[i]
        per_cell = 0
        for v in range(-1, len(v_list)-1):
            vert1 = vertices_pos[v_list[v]]-centroids[i]
            vert2 = vertices_pos[v_list[v+1]]-centroids[i]
            cross_product = vert1[0] * vert2[1] - vert1[1] * vert2[0]
            per_cell += torch.abs(cross_product)/2

        areas.append(per_cell)

    areas = torch.stack(areas)

    return areas

def get_voronoi_perimeters(vertices_pos, vertices_per_cell, device):
    perimeters = []
    for v_list in vertices_per_cell:

        per_cell = 0
        for v in range(-1, len(v_list)-1):
            v1 = vertices_pos[v_list[v]]
            v2 = vertices_pos[v_list[v+1]]
            per_cell += torch.dist(v1, v2)

        perimeters.append(per_cell)

    perimeters = torch.stack(perimeters)

    return perimeters

def get_voronoi_lengths(vertices_pos, vertices_per_cell, device):

    lengths = []
    for v_list in vertices_per_cell:

        per_cell = []
        for v in range(-1, len(v_list)-1):
            v1 = vertices_pos[v_list[v]]
            v2 = vertices_pos[v_list[v+1]]
            per_cell.append(torch.dist(v1, v2))

        lengths.append(per_cell)

    return lengths

def get_voronoi_centroids(vertices_pos, vertices_per_cell, device):

    centroids = []
    for v_list in vertices_per_cell:
        centroids.append(torch.mean(vertices_pos[v_list],dim=0))

    centroids = torch.stack(centroids)

    return centroids

def cell_energy(voronoi_area, voronoi_perimeter, voronoi_lengths, device):

    energy = []
    return energy



def calculate_volume(segmented_image, voxel_size=(0.75, 0.75, 1.0)):
    print("Starting volume calculation...")
    start_time = time.time()

    labels, counts = np.unique(segmented_image, return_counts=True)
    valid_mask = (labels != 0)
    labels = labels[valid_mask]
    counts = counts[valid_mask]
    voxel_volume = np.prod(voxel_size)
    volume_by_label = {label: count * voxel_volume for label, count in zip(labels, counts)}

    volume_df = pd.DataFrame.from_dict(volume_by_label, orient='index', columns=['volume'])

    end_time = time.time()
    print(f"Volume calculation completed in {end_time - start_time:.2f} seconds.")
    return volume_df

def calculate_surface_area_from_segmented_image(segmented_image, voxel_size=(0.75, 0.75, 1.0), smoothing_iterations=2, n_jobs=-1):
    print("Starting surface area calculation...")
    start_time = time.time()

    vol = Volume(segmented_image, spacing=voxel_size)
    labels = np.unique(segmented_image)
    labels = labels[labels != 0]

    mesh = vol.isosurface_discrete(values=labels,
                                   background_label=0,
                                   internal_boundaries=True,
                                   nsmooth=smoothing_iterations)

    vertices = mesh.vertices
    cells = np.array(mesh.cells)
    labels_on_cells = mesh.celldata['BoundaryLabels']

    # Compute the areas of all triangles first
    v0, v1, v2 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
    edge1, edge2 = v1 - v0, v2 - v0
    cross_prod = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Parallelize only the summation of areas for each label
    def sum_areas_for_label(label):
        return label, np.sum(areas[(labels_on_cells[:, 0] == label) | (labels_on_cells[:, 1] == label)])

    results = Parallel(n_jobs=n_jobs)(
        delayed(sum_areas_for_label)(label) for label in labels
    )

    area_df = pd.DataFrame(results, columns=['label', 'surface_area']).set_index('label')

    end_time = time.time()
    print(f"Surface area calculation completed in {end_time - start_time:.2f} seconds.")
    return area_df

def generate_and_save_mesh(segmented_image, voxel_size, output_folder_base, base_name, smoothing_iterations=2):
    print("Starting mesh generation and saving...")
    start_time = time.time()

    vol = Volume(segmented_image, spacing=voxel_size)
    labels = np.unique(segmented_image)
    labels = labels[labels != 0]

    mesh = vol.isosurface_discrete(values=labels,
                                   background_label=0,
                                   internal_boundaries=True,
                                   nsmooth=smoothing_iterations)

    # Extract vertices and cells
    vertices = mesh.vertices
    cells = np.array(mesh.cells)
    labels_on_cells = mesh.celldata['BoundaryLabels']

    # Save mesh as .vtp
    vtp_output_folder = output_folder_base + "_mesh_vtp"
    os.makedirs(vtp_output_folder, exist_ok=True)
    vtp_output_path = os.path.join(vtp_output_folder, base_name + ".vtp")
    mesh.write(vtp_output_path)

    # Save vertices and faces to CSV files
    csv_output_folder = output_folder_base + "_mesh_csv"
    os.makedirs(csv_output_folder, exist_ok=True)

    vertices_df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    vertices_csv_path = os.path.join(csv_output_folder, base_name + "_vertices.csv")
    vertices_df.to_csv(vertices_csv_path, index=False)

    faces_df = pd.DataFrame(cells, columns=['v1', 'v2', 'v3'])
    faces_csv_path = os.path.join(csv_output_folder, base_name + "_faces.csv")
    faces_df.to_csv(faces_csv_path, index=False)

    labels_df = pd.DataFrame(labels_on_cells, columns=['label1', 'label2'])
    labels_csv_path = os.path.join(csv_output_folder, base_name + "_labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)

    end_time = time.time()
    print(f"Mesh generation and saving completed in {end_time - start_time:.2f} seconds.")

    return vertices, cells, labels, labels_on_cells

def calculate_surface_area(vertices, cells, labels, labels_on_cells, n_jobs=-1):
    print("Starting surface area calculation...")
    start_time = time.time()

    # Compute the areas of all triangles first
    v0, v1, v2 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
    edge1, edge2 = v1 - v0, v2 - v0
    cross_prod = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Parallelize only the summation of areas for each label
    def sum_areas_for_label(label):
        return label, np.sum(areas[(labels_on_cells[:, 0] == label) | (labels_on_cells[:, 1] == label)])

    results = Parallel(n_jobs=n_jobs)(
        delayed(sum_areas_for_label)(label) for label in labels
    )

    area_df = pd.DataFrame(results, columns=['label', 'surface_area']).set_index('label')

    end_time = time.time()
    print(f"Surface area calculation completed in {end_time - start_time:.2f} seconds.")
    return area_df

def calculate_centroids_and_accumulators(segmented_image, voxel_size=(0.75, 0.75, 1.0)):
    print("Starting centroid and accumulator calculation...")
    start_time = time.time()

    labels = np.unique(segmented_image)
    labels = labels[labels != 0]

    label_max = labels.max()
    sum_z = np.zeros(label_max + 1, dtype=np.float64)
    sum_y = np.zeros(label_max + 1, dtype=np.float64)
    sum_x = np.zeros(label_max + 1, dtype=np.float64)
    counts = np.zeros(label_max + 1, dtype=np.int64)
    covariance_matrix = np.zeros((label_max + 1, 3, 3), dtype=np.float64)

    # Get the coordinates of all non-zero voxels
    coords = np.column_stack(np.nonzero(segmented_image))
    values = segmented_image[segmented_image > 0]

    # Accumulate the coordinates and counts for each label
    scaled_coords = coords * voxel_size
    np.add.at(sum_z, values, scaled_coords[:, 2])
    np.add.at(sum_y, values, scaled_coords[:, 1])
    np.add.at(sum_x, values, scaled_coords[:, 0])
    np.add.at(counts, values, 1)

    # Compute the centroids
    centroids_z = sum_z[labels] / counts[labels]
    centroids_y = sum_y[labels] / counts[labels]
    centroids_x = sum_x[labels] / counts[labels]

    centroids = np.column_stack((centroids_x, centroids_y, centroids_z))

    # Fill the covariance matrices
    for i in range(3):
        for j in range(i, 3):
            np.add.at(covariance_matrix[:, i, j], values, scaled_coords[:, i] * scaled_coords[:, j])
            if i != j:
                covariance_matrix[:, j, i] = covariance_matrix[:, i, j]

    end_time = time.time()
    print(f"Centroid and accumulator calculation completed in {end_time - start_time:.2f} seconds.")
    return centroids, counts, covariance_matrix, labels

def calculate_elongation_and_orientation(centroids, counts, covariance_matrix, labels):
    print("Starting optimized elongation and orientation calculation...")
    start_time = time.time()

    elongation_results = []
    for label, cov_matrix in zip(labels, covariance_matrix[labels]):
        if counts[label] < 3:
            continue
        centroid = centroids[labels == label]
        # Adjust covariance matrices to be mean-centered
        cov_matrix -= np.outer(centroid, centroid) * counts[label]

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        elongation = np.sqrt(eigenvalues[-1]) / np.sqrt(eigenvalues[0])
        principal_axis = eigenvectors[:, -1]
        elongation_results.append((label, elongation, *principal_axis))

    elongation_df = pd.DataFrame(elongation_results, columns=['label', 'elongation', 'eigenvector_x', 'eigenvector_y', 'eigenvector_z']).set_index('label')

    end_time = time.time()
    print(f"Elongation and orientation calculation completed in {end_time - start_time:.2f} seconds.")
    return elongation_df

def compute_intensity_statistics(raw_image, segmented_image):
    print("Starting intensity statistics computation...")
    start_time = time.time()

    labels = np.unique(segmented_image)

    assert labels[0] == 0, "Background label '0' is missing from the segmented image."

    flat_raw = raw_image.ravel()
    flat_labels = segmented_image.ravel()

    sum_intensity = np.bincount(flat_labels, weights=flat_raw)
    count_intensity = np.bincount(flat_labels)

    mean_intensity = sum_intensity / count_intensity

    sum_squared_diff = np.bincount(flat_labels, weights=(flat_raw - mean_intensity[flat_labels])**2)

    stddev_intensity = np.sqrt(sum_squared_diff / count_intensity)

    background_mean_intensity = mean_intensity[0]

    adjusted_mean_intensity = mean_intensity - background_mean_intensity

    labels = labels[1:]
    adjusted_mean_intensity = adjusted_mean_intensity[1:]
    stddev_intensity = stddev_intensity[1:]
    snr = adjusted_mean_intensity / stddev_intensity

    df = pd.DataFrame({
        'label': labels,
        'mean_intensity': adjusted_mean_intensity,
        'stddev': stddev_intensity,
        'snr': snr
    }).set_index('label')

    end_time = time.time()
    print(f"Intensity statistics computation completed in {end_time - start_time:.2f} seconds.")
    return df


def visualize_mesh(mesh_file):
    # Load the mesh file
    mesh = vedo.load(mesh_file)
    mesh.color('#00FF00')

    # Display the mesh
    plotter = vedo.Plotter()
    plotter.show(mesh, "3D Mesh Visualization",
                 axes=4,
                 bg='black',
                 azimuth=0,
                #  elevation=180,
                #  roll=180,
                #  viewup="z"
                 )





def process_image_batch(segmented_image_path, raw_image_path,
         voxel_size=(0.75, 0.75, 1.0), smoothing_iterations=2, n_jobs=-1,
         mesh_only=False, props_only=False):
    start_time_all = time.time()

    print("Reading images...")
    start_time = time.time()

    # note that vedo use a default xyz order for arrays,
    # which is opposite to the zyx order of skimage.io.imread()
    segmented_image = io.imread(segmented_image_path)
    segmented_image = np.swapaxes(segmented_image, 0, 2)

    if not mesh_only:
        raw_image = io.imread(raw_image_path)
        raw_image = np.swapaxes(raw_image, 0, 2)

    end_time = time.time()
    print(f"Images read in {end_time - start_time:.2f} seconds.")

    # Parse out the output folder base and base name
    parent_dir = os.path.dirname(os.path.dirname(segmented_image_path))
    base_name = os.path.splitext(os.path.basename(segmented_image_path))[0]
    output_folder_base = os.path.join(parent_dir, os.path.basename(os.path.dirname(segmented_image_path)) + '_smooth' + str(smoothing_iterations))

    if not props_only:
        # Generate mesh, save to .vtp and CSV
        vertices, cells, labels, labels_on_cells = generate_and_save_mesh(segmented_image, voxel_size, output_folder_base, base_name, smoothing_iterations)

    if not mesh_only:
        # calculate surface area
        if props_only:
            surface_area_df = calculate_surface_area_from_segmented_image(segmented_image, voxel_size, smoothing_iterations, n_jobs)
        else:
            surface_area_df = calculate_surface_area(vertices, cells, labels, labels_on_cells, n_jobs=n_jobs)

        # Compute morphometrics
        centroids, counts, covariance_matrix, labels = calculate_centroids_and_accumulators(segmented_image, voxel_size)
        centroid_df = pd.DataFrame(centroids, index=labels, columns=['centroid_x', 'centroid_y', 'centroid_z'])
        elongation_df = calculate_elongation_and_orientation(centroids, counts, covariance_matrix, labels)
        volume_df = calculate_volume(segmented_image, voxel_size)

        morphometrics_df = pd.concat([volume_df, surface_area_df, centroid_df, elongation_df], axis=1)
        morphometrics_df['sphericity'] = (np.pi**(1/3) * (6 * morphometrics_df['volume'])**(2/3)) / morphometrics_df['surface_area']

        # Compute fluorescence stats
        fluorescence_df = compute_intensity_statistics(raw_image, segmented_image)

        # Merge DataFrames
        final_df = pd.concat([morphometrics_df, fluorescence_df], axis=1)

        # Filter out rows with inf or nan values
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_df.dropna(inplace=True)

        # Reset index to ensure label is saved as a column
        final_df.reset_index(inplace=True)

        # Rename index to label
        final_df.rename(columns={'index': 'label'}, inplace=True)

        # Determine output folder and file path for CSV
        output_folder = output_folder_base + '_label_props'
        os.makedirs(output_folder, exist_ok=True)

        # Output file name
        output_file_name = base_name + '_label_props.csv'
        output_file = os.path.join(output_folder, output_file_name)

        # Save to CSV
        start_time = time.time()
        final_df.to_csv(output_file, index=False)
        end_time = time.time()
        print(f"Results saved to {output_file} in {end_time - start_time:.2f} seconds.")

    end_time_all = time.time()
    print(f"Total processing time: {end_time_all - start_time_all:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute morphometrics and fluorescence stats from 3D images.")
    parser.add_argument("segmented_image_path", type=str, help="Path to the segmented image file (TIFF).")
    parser.add_argument("raw_image_path", type=str, help="Path to the raw image file (TIFF).")
    parser.add_argument("-vs", "--voxel_size", type=float, nargs=3, default=(0.75, 0.75, 1.0), help="Voxel size as three floats (x, y, z), e.g., -vs 0.75 0.75 1.0")
    parser.add_argument("-si", "--smoothing_iterations", type=int, default=2, help="Number of smoothing iterations for surface area calculation. Default is 2.")
    parser.add_argument("-n", "--n_jobs", type=int, default=-1, help="Number of parallel jobs to use. Default is -1 (use all processors).")
    parser.add_argument('--mesh_only', action='store_true', help='Only compute and save mesh (not label properties)')
    parser.add_argument('--props_only', action='store_true', help='Only compute and save label properties (not saving mesh)')

    args = parser.parse_args()
    process_image_batch(args.segmented_image_path, args.raw_image_path, tuple(args.voxel_size), smoothing_iterations=args.smoothing_iterations, n_jobs=args.n_jobs, mesh_only=args.mesh_only, props_only=args.props_only)

    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualize_mesh.py <mesh_file>")
        sys.exit(1)

    mesh_file = sys.argv[1]
    visualize_mesh(mesh_file)













# fig, ax = fig_init()
# voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
#                 point_size=0)
# plt.scatter(points[:, 0], points[:, 1], s=30, color='blue')
# plt.scatter(vertices[:, 0], vertices[:, 1], s=30, color='green')
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])