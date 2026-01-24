import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D(pyg.nn.MessagePassing):
    """
    Computes interactions between particles and fields, and between particles.
    Implements diffusiophoresis with short-range repulsion.
    Supports multiple particle types with different mobilities and consumption/production rates.

    Per-type parameters (from params array, one row per type):
        params[type_id] = [M_magnitude, consumption_rate, production_rate, ...]

    Global parameters (from params_mesh):
        params_mesh[2, 0] = Pe (Péclet number)
        params_mesh[2, 3] = influence_radius
    """

    def __init__(self, aggr_type='mean', p=None, p_mesh=None, bc_dpos=None, dimension=2):
        super(PDE_D, self).__init__(aggr=aggr_type)

        self.p = p  # Per-type parameters: [n_types, n_params]
        self.p_mesh = p_mesh  # Global mesh parameters
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.n_particle_types = p.shape[0] if p is not None else 1

        # Global parameters from p_mesh
        if p_mesh is not None:
            self.Pe = p_mesh[2, 0]
            self.influence_radius = p_mesh[2, 3]
        else:
            self.Pe = torch.tensor(0.1)
            self.influence_radius = torch.tensor(0.05)

        # Particle-particle repulsion parameters (same for all types)
        self.repulsion_strength = 50
        self.repulsion_range = 0.025

        # Print per-type parameters
        print(f"Initialized PDE_D with {self.n_particle_types} particle types:")
        for t in range(self.n_particle_types):
            M_mag = torch.abs(p[t, 0]).item()
            consumption = p[t, 1].item() if p.shape[1] > 1 else 0.0
            production = p[t, 2].item() if p.shape[1] > 2 else 0.0
            print(f"  Type {t}: M=±{M_mag:.1f}, consumption={consumption:.1f}, production={production:.1f}")
        print(f"Global: Pe={self.Pe.item():.3f}, influence_radius={self.influence_radius.item():.3f}")
    
    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Get particle types and per-type parameters (consistent with PDE_A pattern)
        particle_type = x[:, 1 + 2 * self.dimension].long()  # x[:, 5] for 2D
        parameters = self.p[to_numpy(particle_type), :]  # Index by particle type

        if direction == 'interpolate':
            # Step 1: Interpolate fields from mesh to particles
            result = self.propagate(edge_index, x=x, mode='interpolate', parameters=parameters)
            return result
        elif direction == 'fp':
            # Step 2: Calculate diffusiophoretic velocities
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)
            return result
        elif direction == 'pf':
            # Particle → Field effects
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            # Particle → Particle repulsion
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result
    
    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode (consistent with PDE_A pattern).

        Per-type parameters (parameters_i, indexed by receiver particle type):
            [:, 0] = M_magnitude (mobility strength)
            [:, 1] = consumption_rate
            [:, 2] = production_rate
            [:, 3] = interaction_strength (for pp mode, can be negative for attraction)
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Interpolate fields from mesh nodes to particles
            # x_j are mesh nodes (senders), x_i are particles (receivers)

            # Get field values from mesh
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]

            # Distance-based weights (inverse distance or Gaussian)
            weight = torch.exp(-dist / 0.01).unsqueeze(1)  # Gaussian kernel

            # Return weighted fields for aggregation
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # Field differences
            fields_i = x_i[:, 6:8]  # Particle fields [C₁, C₂]
            fields_j = x_j[:, 6:8]  # Mesh fields [C₁, C₂]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            # Use smoothing kernel to avoid sharp gradients
            kernel = torch.exp(-dist / 0.05)  # Smoothing length scale

            # Direction vector
            dir_norm = d_pos / dist_safe.unsqueeze(1)

            # Smooth gradient estimation
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            # Per-type mobility (M1 positive, M2 negative - opposite signs enforced)
            M_mag = torch.abs(parameters_i[:, 0:1])  # Mobility magnitude per particle
            M1 = M_mag   # Positive for C1
            M2 = -M_mag  # Negative for C2 (opposite sign)

            # Diffusiophoretic velocity with per-type mobility
            velocities = (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            return velocities

        elif mode == 'pf':
            # Particle → Field: Calculate field updates with per-type rates
            # Gaussian influence based on distance
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            # Per-type consumption and production rates
            consumption = parameters_i[:, 1] if parameters_i.shape[1] > 1 else torch.zeros_like(weights)
            production = parameters_i[:, 2] if parameters_i.shape[1] > 2 else torch.zeros_like(weights)

            # Create field updates [C₁, C₂]
            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle → Particle: Type-dependent interaction
            # Positive = repulsion, Negative = attraction
            forces = torch.zeros_like(pos_i)

            # Apply force only for particles within range
            in_range = dist < self.repulsion_range
            if in_range.any():
                # Direction vectors
                dir_norm = d_pos / dist_safe.unsqueeze(1)

                # Get interaction strength from parameters (column 3)
                # Default to repulsion_strength if not specified
                if parameters_i.shape[1] > 3:
                    interaction = parameters_i[in_range, 3]
                else:
                    interaction = torch.full((in_range.sum(),), self.repulsion_strength,
                                            device=pos_i.device, dtype=pos_i.dtype)

                # Interaction magnitude (exponential decay)
                # Positive interaction = repulsion, Negative = attraction
                interaction_mag = interaction * torch.exp(
                    -5.0 * dist[in_range] / self.repulsion_range
                )

                # Apply forces (negative sign for repulsion convention)
                forces[in_range] = -dir_norm[in_range] * interaction_mag.unsqueeze(1)

            return forces
    
    def update(self, aggr_out, mode=None):
        """
        Process aggregated messages
        """
        if mode == 'interpolate':
            # Normalize weighted average
            C1_weighted = aggr_out[:, 0:1]
            C2_weighted = aggr_out[:, 1:2]
            weight_sum = aggr_out[:, 2:3]
            
            # Avoid division by zero
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            
            # Return normalized fields
            return torch.cat([C1_weighted / weight_sum, C2_weighted / weight_sum], dim=1)
        else:
            return aggr_out