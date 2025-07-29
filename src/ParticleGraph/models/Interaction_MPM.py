import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy, reparameterize
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.models.Gumbel import gumbel_softmax_sample, gumbel_softmax
from ParticleGraph.generators.MPM_P2G import MPM_P2G
from ParticleGraph.utils import *
import torch_geometric.data as data

class Interaction_MPM(nn.Module):

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_MPM, self).__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.model = model_config.particle_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim

        self.input_size_nnr = model_config.input_size_nnr
        self.n_layers_nnr = model_config.n_layers_nnr
        self.hidden_dim_nnr = model_config.hidden_dim_nnr
        self.output_size_nnr = model_config.output_size_nnr
        self.outermost_linear_nnr = model_config.outermost_linear_nnr
        self.omega= model_config.omega


        self.n_particle_types = simulation_config.n_particle_types
        self.n_particles = simulation_config.n_particles
        self.n_grid = simulation_config.n_grid

        self.delta_t = simulation_config.delta_t
        self.n_frames = simulation_config.n_frames
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.grid_i, self.grid_j = torch.meshgrid(
            torch.arange(self.n_grid, device=device, dtype=torch.float32),
            torch.arange(self.n_grid, device=device, dtype=torch.float32),
            indexing='ij'
        )  # Shape: [n_grid, n_grid]
        self.grid_coords = self.dx * torch.stack([
            self.grid_i,  # x coordinates
            self.grid_j  # y coordinates
        ], dim=-1).reshape(-1, 2)  # Shape: [1024, 2]

        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.offsets = torch.tensor([[i, j] for i in range(3) for j in range(3)],
                               device=device, dtype=torch.float32)  # [9, 2]
        self.particle_offsets = self.offsets.unsqueeze(0).expand(self.n_particles, -1, -1)
        self.expansion_factor = simulation_config.MPM_expansion_factor
        self.gravity = simulation_config.MPM_gravity



        self.model_MPM = MPM_P2G(aggr_type='add', device=device)

        self.siren_F_Jp = Siren(in_features=3, out_features=4, hidden_features=self.hidden_dim_nnr,
                           hidden_layers=self.n_layers_nnr, first_omega_0=self.omega, hidden_omega_0=self.omega, outermost_linear=True).to(device)

        self.siren_C = Siren(in_features=5, out_features=4, hidden_features=self.hidden_dim_nnr,
                           hidden_layers=self.n_layers_nnr, first_omega_0=self.omega, hidden_omega_0=self.omega, outermost_linear=True).to(device)

        self.MLP_mu_lambda = MLP(input_size=self.embedding_dim + 1, output_size=2, nlayers=3, hidden_size=32, device=device)
        self.MLP_sig_plastic_ratio = MLP(input_size=self.embedding_dim + 12, output_size=3, nlayers=5, hidden_size=128, device=device)
        self.MLP_F = MLP(input_size=self.embedding_dim + 13, output_size=4, nlayers=5, hidden_size=128, device=device)

        # self.mlp0 = MLP(input_size=3, output_size=1, nlayers=5, hidden_size=128, device=device)
        # self.mlp1 = MLP(input_size=2, output_size=1, nlayers=2, hidden_size=4, device=device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))


    def forward(self, data=[], data_id=[], k=[], trainer=[]):

        x = data.x
        self.data_id = data_id.long()

        N = x[:,0:1]
        pos = x[:,1:3]  # pos is the absolute position
        d_pos = x[:,3:5]  # d_pos is the velocity
        C = x[:,5:9].reshape(-1, 2, 2)  # C is the affine deformation gradient
        F = x[:,9:13].reshape(-1, 2, 2) # F is the deformation gradient
        T = x[:,13:14].long() # T is the type of particle
        Jp = x[:,14:15] # Jp is the Jacobian of the deformation gradient
        M = x[:,15:16]  # M is the mass of the particle
        S = x[:,16:20].reshape(-1, 2, 2)
        frame = k / self.n_frames

        embedding = self.a[self.data_id.detach(), N.long(), :].squeeze()

        if 'C' in trainer:
            features = torch.cat((pos, d_pos, frame), dim=1).detach()
            C_sample = self.siren_C(features).reshape(-1, 2, 2)
            return C_sample.reshape(-1, 4)

        elif trainer == 'F':
            features = torch.cat((pos, frame), dim=1).detach()
            F_sample = self.siren_F_Jp(features)[:,0:4]
            return F_sample.reshape(-1, 4)

        elif trainer == 'next_F':
            features = torch.cat((pos, frame), dim=1).detach()
            F_sample = self.siren_F_Jp(features).reshape(-1, 2, 2)
            X_, V_, C_, F_, T_, Jp_, M_, S_, GM_, GV_ = self.MPM_engine(self.model_MPM, pos, d_pos, C, F_sample,
                                                        T, Jp, M, self.n_particles, self.n_grid,
                                                       self.delta_t, self.dx, self.inv_dx, embedding,
                                                       self.offsets, self.particle_offsets, self.grid_coords,
                                                       self.expansion_factor, self.gravity, self.device)
            return F_


    def MPM_engine(self, model_MPM, X, V, C, F, T, Jp, M, n_particles,
                       n_grid, dt, dx, inv_dx, embedding,
                       offsets, particle_offsets, grid_coords, expansion_factor,
                        gravity, device):
        """
        MPM substep implementation
        """

        # Initialize
        p_mass = M.squeeze(-1)
        grid_m = torch.zeros((n_grid, n_grid), device=device, dtype=torch.float32)
        grid_v = torch.zeros((n_grid, n_grid, 2), device=device, dtype=torch.float32)

        # Create identity matrices for all particles
        identity = torch.eye(2, device=device).unsqueeze(0).expand(X.shape[0], -1, -1)

        # Calculate F ############################################################################################

        # Update deformation gradient: F = (I + dt * C) * F_old
        F = (identity + dt * C) @ F

        # Hardening coefficient
        h = torch.exp(10 * (1.0 - Jp.squeeze()))

        in_features = torch.cat((embedding, h[:,None]), dim=1)
        out1 = self.MLP_mu_lambda(in_features)
        mu = out1[:, 0]**2  # Shear modulus >0
        la = out1[:, 1]  # Lame's first parameter

        # SVD decomposition
        U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')

        in_features = torch.cat((embedding, U.reshape(-1, 4), sig, Vh.reshape(-1, 4),
                                 torch.det(U)[:,None], torch.det(Vh)[:,None]), dim=1)
        out2 = self.MLP_sig_plastic_ratio(in_features)
        sig = out2[:, 0:2]**2 # Singular values
        plastic_ratio = out2[:, 2:3]**2  # Plastic ratio

        Jp = Jp * plastic_ratio
        J = torch.prod(sig, dim=1)
        sig_diag = torch.diag_embed(sig)

        in_features = torch.cat((embedding, J[:,None], U.reshape(-1, 4), sig_diag.reshape(-1, 4), Vh.reshape(-1, 4)), dim=1)
        F = self.MLP_F(in_features)

        return X, V, C, F, T, Jp, M, F, grid_m, grid_v

        F[liquid_mask] = F_liquid[liquid_mask]
        solid_mask = jelly_mask | snow_mask
        F[solid_mask] = F_solid[solid_mask]

        # Calculate stress ############################################################################################
        R = U @ Vh
        F_minus_R = F - R
        stress = (2 * mu.unsqueeze(-1).unsqueeze(-1) * F_minus_R @ F.transpose(-2, -1) +
                  identity * (la * J * (J - 1)).unsqueeze(-1).unsqueeze(-1))
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass.unsqueeze(-1).unsqueeze(-1) * C

        # P2G loop ###################################################################################################

        base = (X * inv_dx - 0.5).int()
        grid_positions = base.unsqueeze(1) + offsets.unsqueeze(0)  # [n_particles, 9, 2]
        particle_indices = torch.arange(n_particles, device=device).unsqueeze(1).expand(-1, 9).flatten()
        grid_indices = grid_positions.flatten().reshape(-1, 2)  # Flatten to [n_particles*9, 2]
        grid_indices_1d = grid_indices[:, 0] * n_grid + grid_indices[:, 1]
        edge_index = torch.stack([particle_indices, grid_indices_1d], dim=0).long()
        edge_index[0, :] += n_grid ** 2  # offset particle indices

        fx = X * inv_dx - base.float()
        fx_per_edge = fx.unsqueeze(1).expand(-1, 9, -1).flatten(end_dim=1)  # [n_particles*9, 2]

        # Compute dpos for each edge (needed for affine contribution)
        particle_offsets_expanded = offsets.unsqueeze(0).expand(n_particles, -1, -1)  # [n_particles, 9, 2]
        particle_fx_expanded = fx.unsqueeze(1).expand(-1, 9, -1)  # [n_particles, 9, 2]
        dpos = (particle_offsets_expanded - particle_fx_expanded) * dx  # [n_particles, 9, 2]
        dpos_per_edge = dpos.flatten(end_dim=1)  # [n_particles*9, 2]

        # Affine matrices per edge (replicate for each particle's 9 edges)
        affine_per_edge = affine.unsqueeze(1).expand(-1, 9, -1, -1).flatten(end_dim=1)  # [n_particles*9, 2, 2]

        # Extended node features: [mass, vel_x, vel_y] for particles, [0, 0, 0] for grid
        grid_features = torch.zeros((n_grid ** 2, 3), dtype=torch.float32, device=device)  # [mass, vel_x, vel_y]
        particle_features = torch.cat([p_mass[:, None], V], dim=1)  # [n_particles, 3]
        x_ = torch.cat([grid_features, particle_features], dim=0)  # [n_grid**2 + n_particles, 3]

        dataset = data.Data(x=x_, edge_index=edge_index, fx_per_edge=fx_per_edge,
                            affine_per_edge=affine_per_edge, dpos_per_edge=dpos_per_edge)
        grid_output = model_MPM(dataset)[0:n_grid ** 2]  # [n_grid**2, 3]
        grid_m = grid_output[:, 0].view(n_grid, n_grid)  # Mass component
        grid_v = grid_output[:, 1:3].view(n_grid, n_grid, 2)  # Velocity components # Reshape to [n_grid, n_grid]

        # Quadratic B-spline kernel weights
        w_0 = 0.5 * (1.5 - fx) ** 2
        w_1 = 0.75 - (fx - 1) ** 2
        w_2 = 0.5 * (fx - 0.5) ** 2
        # Stack weights [n_particles, 3, 2]
        w = torch.stack([w_0, w_1, w_2], dim=1)

        # Create mask for valid grid points (non-zero mass)
        valid_mass_mask = grid_m > 0
        # Convert momentum to velocity (vectorized)
        grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                             grid_v / grid_m.unsqueeze(-1),
                             grid_v)

        # Apply gravity (vectorized)
        gravity_force = torch.tensor([0.0, dt * (gravity)], device=device)
        grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                             grid_v + gravity_force,
                             grid_v)

        # VECTORIZED Boundary conditions
        # Create coordinate grids for boundary checking
        i_coords = torch.arange(n_grid, device=device).unsqueeze(1).expand(n_grid, n_grid)  # [n_grid, n_grid]
        j_coords = torch.arange(n_grid, device=device).unsqueeze(0).expand(n_grid, n_grid)  # [n_grid, n_grid]
        # Left boundary: i < 3 and v_x < 0 → set v_x = 0
        left_boundary_mask = (i_coords < 3) & (grid_v[:, :, 0] < 0) & valid_mass_mask
        grid_v[:, :, 0] = torch.where(left_boundary_mask, 0.0, grid_v[:, :, 0])
        # Right boundary: i > n_grid - 3 and v_x > 0 → set v_x = 0
        right_boundary_mask = (i_coords > n_grid - 3) & (grid_v[:, :, 0] > 0) & valid_mass_mask
        grid_v[:, :, 0] = torch.where(right_boundary_mask, 0.0, grid_v[:, :, 0])
        # Bottom boundary: j < 3 and v_y < 0 → set v_y = 0
        bottom_boundary_mask = (j_coords < 3) & (grid_v[:, :, 1] < 0) & valid_mass_mask
        grid_v[:, :, 1] = torch.where(bottom_boundary_mask, 0.0, grid_v[:, :, 1])
        # Top boundary: j > n_grid - 3 and v_y > 0 → set v_y = 0
        top_boundary_mask = (j_coords > n_grid - 3) & (grid_v[:, :, 1] > 0) & valid_mass_mask
        grid_v[:, :, 1] = torch.where(top_boundary_mask, 0.0, grid_v[:, :, 1])

        # G2P transfer - CORRECTED VERSION
        new_V = torch.zeros_like(V)
        new_C = torch.zeros_like(C)

        # G2P loop ###################################################################################################
        # Process all 9 neighbors simultaneously (using pre-computed offsets)

        # Expand offset for all particles and compute dpos for all neighbors (using pre-computed fx)
        dpos_all = offsets.unsqueeze(0) - fx.unsqueeze(1)  # [n_particles, 9, 2]

        # Grid positions for all neighbors (using pre-computed base)
        grid_pos_all = base.unsqueeze(1) + offsets.long().unsqueeze(0)  # [n_particles, 9, 2]

        # Weights for all neighbors: w[:, i, 0] * w[:, j, 1] for all (i,j) combinations (using pre-computed w)
        i_indices = offsets[:, 0].long()  # [9] - i values: [0,0,0,1,1,1,2,2,2]
        j_indices = offsets[:, 1].long()  # [9] - j values: [0,1,2,0,1,2,0,1,2]
        weights_all = w[:, i_indices, 0] * w[:, j_indices, 1]  # [n_particles, 9]

        # Bounds checking for all neighbors
        valid_mask_all = ((grid_pos_all[:, :, 0] >= 0) & (grid_pos_all[:, :, 0] < n_grid) &
                          (grid_pos_all[:, :, 1] >= 0) & (grid_pos_all[:, :, 1] < n_grid))  # [n_particles, 9]

        # Get grid velocities for all neighbors with bounds checking
        g_v_all = torch.zeros((n_particles, 9, 2), device=device)

        # Flatten for efficient indexing
        flat_valid = valid_mask_all.flatten()  # [n_particles * 9]
        flat_grid_pos = grid_pos_all.reshape(-1, 2)  # [n_particles * 9, 2]

        if flat_valid.any():
            valid_positions = flat_grid_pos[flat_valid]
            flat_g_v = torch.zeros((n_particles * 9, 2), device=device)
            flat_g_v[flat_valid] = grid_v[valid_positions[:, 0], valid_positions[:, 1]]
            g_v_all = flat_g_v.reshape(n_particles, 9, 2)

        # Accumulate velocity contributions from all neighbors
        velocity_contribs = weights_all.unsqueeze(-1) * g_v_all  # [n_particles, 9, 2]
        new_V += velocity_contribs.sum(dim=1)  # Sum over the 9 neighbors

        # CORRECTED APIC update - vectorized outer product for all neighbors
        # Reshape for batch matrix multiplication: [n_particles * 9, 2, 1] x [n_particles * 9, 1, 2]
        g_v_flat = g_v_all.reshape(-1, 2, 1)  # [n_particles * 9, 2, 1]
        dpos_flat = dpos_all.reshape(-1, 1, 2)  # [n_particles * 9, 1, 2]
        outer_products = torch.bmm(g_v_flat, dpos_flat).reshape(n_particles, 9, 2, 2)  # [n_particles, 9, 2, 2]

        # Weight the outer products and sum over neighbors
        weighted_outer_products = weights_all.unsqueeze(-1).unsqueeze(-1) * outer_products  # [n_particles, 9, 2, 2]
        new_C += 4 * inv_dx * weighted_outer_products.sum(dim=1)  # Sum over the 9 neighbors

        # Particle advection
        X = X + dt * new_V

        F = F.reshape(n_particles, 4)
        return X, new_V, new_C, F, T, Jp, M, F, grid_m, grid_v

        return X, V, C, F, T, Jp, M, stress, grid_m, grid_v


