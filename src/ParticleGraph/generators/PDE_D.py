import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D(pyg.nn.MessagePassing):
    """
    Computes interactions between particles and fields, and between particles.
    Implements diffusiophoresis with PDE_A-style attraction-repulsion.

    Supports multiple particle types when particle_params is provided.
    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
      - M1, M2: Mobility coefficients for diffusiophoresis
      - consumption, production: Particle effects on fields
      - ar_p1, ar_p2: Attraction term parameters (p1*exp(-d^(2*p2)/(2σ²)))
      - ar_p3, ar_p4: Repulsion term parameters (p3*exp(-d^(2*p4)/(2σ²)))
    """

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D, self).__init__(aggr=aggr_type)

        self.p = p  # Mesh params (shared across all types)
        self.particle_params = particle_params  # Per-type particle params (optional)
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma  # For attraction-repulsion kernel

        # Global parameters from mesh (used as fallback when particle_params=None)
        # Diffusiophoretic parameters
        self.M1 = p[0, 5]       # Mobility coefficient for C₁
        self.M2 = p[1, 1]       # Mobility coefficient for C₂

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Péclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters
        # Block 2 code change: increased range 0.025->0.04 to encourage spreading
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Block 8 code change: Gradient boost for nonlinear mobility response
        # p[2, 4] controls boost exponent: 0 = linear, >0 = superlinear response to gradients
        # Hypothesis: Boosting response at steep gradients creates stronger boundary attraction
        # (opposite of saturation which failed - try amplifying instead of limiting)
        self.boost_exponent = p[2, 4] if p.shape[1] > 4 else 0.0

        # Report configuration
        print(f"initialized PDE_D with parameters:")
        print(f"mobility: M₁={self.M1.item()}, M₂={self.M2.item()}")
        if hasattr(self, 'boost_exponent'):
            boost_val = self.boost_exponent.item() if hasattr(self.boost_exponent, 'item') else self.boost_exponent
            if boost_val > 0:
                print(f"gradient boost exponent: {boost_val:.3f} (0=linear, >0=superlinear)")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle→Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")
    
    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available (like PDE_A pattern)
        if self.particle_params is not None:
            # Particle type is at index 1 + 2*dimension (after pos_x, pos_y)
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D: particle_params has {n_param_rows} rows but found "
                    f"particle type {max_type}. Add {max_type + 1} rows to "
                    f"simulation.params (one per particle type)."
                )
            parameters = self.particle_params[to_numpy(particle_type), :]
        else:
            parameters = None

        if direction == 'interpolate':
            # Step 1: Interpolate fields from mesh to particles
            result = self.propagate(edge_index, x=x, mode='interpolate', parameters=parameters)

            # For out-of-box particles, return zero fields (no valid mesh data)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()

            return result
        elif direction == 'fp':
            # Step 2: Calculate diffusiophoretic velocities
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)

            # Zero out velocities for particles outside [0,1] box
            # These particles have no valid field data to interpolate from
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()

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
        Compute messages based on mode

        Per-type params layout (when parameters_i provided):
        [M1, M2, consumption, production, repulsion_strength, repulsion_range]
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

            # Smooth gradient estimation (not raw difference/distance)
            # Scale by domain size since positions are [0,1] but physics expects [0,32]
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            # Get mobility coefficients (per-type or global)
            if parameters_i is not None:
                M1 = parameters_i[:, 0:1]  # Per-particle type
                M2 = parameters_i[:, 1:2]
            else:
                M1 = self.M1  # Global fallback
                M2 = self.M2

            # Diffusiophoretic velocity (raw linear response)
            velocity_raw = (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            # Block 8: Apply gradient boost if enabled
            # This amplifies response at steep gradients (pattern boundaries)
            # Opposite of saturation which failed - enhances rather than limits
            if hasattr(self, 'boost_exponent') and self.boost_exponent > 0:
                # Compute gradient magnitude
                grad_mag = torch.sqrt(grad_C1**2 + grad_C2**2 + 1e-8)
                # Superlinear boost: multiply by grad_mag^exponent
                # exponent=0.5: sqrt boost (mild), exponent=1.0: linear boost (stronger)
                # This makes particles accelerate toward steep boundaries
                boost_factor = 1.0 + (grad_mag ** self.boost_exponent)
                velocities = velocity_raw * boost_factor
            else:
                velocities = velocity_raw

            return velocities

        elif mode == 'pf':
            # Particle → Field: Calculate field updates
            # Gaussian influence based on distance
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            # Get consumption/production rates (per-type or global)
            if parameters_i is not None:
                consumption = parameters_i[:, 2]
                production = parameters_i[:, 3]
            else:
                consumption = self.consumption_rate
                production = self.production_rate

            # Create field updates [C₁, C₂]
            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle → Particle: PDE_A-style attraction-repulsion
            # Formula: f = (p1 * exp(-d^(2*p2) / (2σ²)) - p3 * exp(-d^(2*p4) / (2σ²))) * direction

            if parameters_i is not None:
                # Per-type attraction-repulsion parameters
                p1 = parameters_i[:, 4]  # ar_p1: attraction strength
                p2 = parameters_i[:, 5]  # ar_p2: attraction exponent
                p3 = parameters_i[:, 6]  # ar_p3: repulsion strength
                p4 = parameters_i[:, 7]  # ar_p4: repulsion exponent

                # PDE_A formula: attraction - repulsion
                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))

                # Apply force in direction of neighbor (attraction positive, repulsion negative)
                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)
            else:
                # Fallback: simple exponential repulsion (backward compatible)
                forces = torch.zeros_like(pos_i)
                in_range = dist < self.repulsion_range
                if in_range.any():
                    dir_norm = d_pos / dist_safe.unsqueeze(1)
                    repulsion_mag = self.repulsion_strength * torch.exp(
                        -5.0 * dist[in_range] / self.repulsion_range
                    )
                    forces[in_range] = -dir_norm[in_range] * repulsion_mag.unsqueeze(1)

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