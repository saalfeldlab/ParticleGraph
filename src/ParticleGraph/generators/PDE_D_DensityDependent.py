import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_DensityDependent(pyg.nn.MessagePassing):
    """
    Density-dependent mobility particle model for diffusiophoresis.

    Implements quorum sensing / contact inhibition of locomotion: particles
    reduce their diffusiophoretic mobility when local particle density is high.
    This creates self-limiting aggregation where clusters stop growing once
    they reach a characteristic density.

    Literature:
    - Bassler (2002) Annual Review of Microbiology 56:63-91
      "Small talk: cell-to-cell communication in bacteria" (quorum sensing)
    - Mayor & Carmona-Fontaine (2010) Trends in Cell Biology 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Cates & Tailleur (2015) Annual Review of Condensed Matter Physics 6:219-244
      "Motility-induced phase separation" (density-dependent motility)

    Physics:
    In standard diffusiophoresis: v = M * nabla_C (constant mobility)
    In density-dependent:         v = M * f(rho) * nabla_C

    where f(rho) = 1 / (1 + (rho/rho_0)^n) is a Hill function that:
    - f(rho) -> 1 when rho << rho_0 (low density: full mobility)
    - f(rho) -> 0 when rho >> rho_0 (high density: mobility suppressed)
    - rho_0 is the critical density threshold
    - n is the Hill coefficient (cooperativity)

    Key differences from linear PDE_D:
    1. Self-limiting aggregation: Dense clusters stop attracting more particles
    2. Size selection: Preferred cluster size set by rho_0
    3. Density waves: Potential for propagating density fronts
    4. Dynamic equilibrium: Clusters maintain steady size even with field gradients

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
      Same as base PDE_D for compatibility.
    """

    PARAMS_DOC = {
        "model_name": "DensityDependent",
        "literature": "Cates & Tailleur (2015) ARCMP 6:219-244; Mayor & Carmona-Fontaine (2010) TCB 20:319-328",
        "description": "Density-dependent mobility: particles slow down at high local density (contact inhibition / quorum sensing)",
        "equations": {
            "field_to_particle": "v = M1 * f(rho) * nabla_C1 + M2 * f(rho) * nabla_C2",
            "density_function": "f(rho) = 1 / (1 + (rho/rho_0)^n), Hill function",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params": [
            {"index": 0, "name": "M1", "description": "Mobility for C1 gradient", "typical_range": [-8, 8]},
            {"index": 1, "name": "M2", "description": "Mobility for C2 gradient", "typical_range": [-8, 8]},
            {"index": 2, "name": "consumption", "description": "C1 consumption rate", "typical_range": [0, 200]},
            {"index": 3, "name": "production", "description": "C2 production rate", "typical_range": [-200, 0]},
            {"index": 4, "name": "ar_p1", "description": "Attraction strength", "typical_range": [0.5, 3.0]},
            {"index": 5, "name": "ar_p2", "description": "Attraction exponent", "typical_range": [0.5, 2.0]},
            {"index": 6, "name": "ar_p3", "description": "Repulsion strength", "typical_range": [0.5, 3.0]},
            {"index": 7, "name": "ar_p4", "description": "Repulsion exponent", "typical_range": [0.5, 2.0]}
        ],
        "density_params": {
            "rho_0": "Critical density threshold (number of neighbors within sensing radius). Default=15. When local count > rho_0, mobility drops.",
            "hill_n": "Hill coefficient (cooperativity). n=1: gradual transition. n=2: sharper switch. n=4: ultrasensitive. Default=2.",
            "sensing_radius": "Radius for counting local neighbors. Default=0.05 (matching pp interaction range).",
            "note": "Effective mobility = M * 1/(1+(count/rho_0)^n). At rho_0 neighbors, mobility = M/2. At 2*rho_0 neighbors, mobility = M/(1+2^n)."
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_DensityDependent, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Global parameters from mesh (used as fallback when particle_params=None)
        self.M1 = p[0, 5]
        self.M2 = p[1, 1]

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Peclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters (same as base PDE_D)
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Density-dependent mobility parameters
        # rho_0: critical neighbor count — when local density reaches this, mobility halves
        # For 9600 particles in [0,1]^2 with sensing radius 0.05:
        # uniform density => ~9600 * pi*0.05^2 = ~75 neighbors on average
        # We want clusters (higher density) to slow down, so rho_0 should be above uniform avg
        # but below the cluster density. rho_0 = 15 is for the pp interaction radius (0.04)
        # where uniform density gives ~9600 * pi*0.04^2 = ~48 neighbors.
        self.rho_0 = p[2, 4] if p.shape[1] > 4 and p[2, 4] != 0 else 15.0
        self.hill_n = p[2, 5] if p.shape[1] > 5 and p[2, 5] != 0 else 2.0
        self.sensing_radius = 0.05  # Same scale as pp interaction

        # Convert to proper tensors if needed
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)

        # Storage for local density (computed in pp pass, used in fp pass)
        self.local_density = None

        # Report configuration
        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        print(f"initialized PDE_D_DensityDependent with parameters:")
        print(f"mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"density-dependent: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius}")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle->Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")

    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction.
        Same interface as base PDE_D for compatibility.

        IMPORTANT: The 'pp' direction must be called BEFORE 'fp' in the simulation loop
        so that local_density is computed before it's needed for mobility modulation.
        This is the standard call order in graph_data_generator.py.
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available
        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D_DensityDependent: particle_params has {n_param_rows} rows but found "
                    f"particle type {max_type}. Add {max_type + 1} rows to "
                    f"simulation.params (one per particle type)."
                )
            parameters = self.particle_params[to_numpy(particle_type), :]
        else:
            parameters = None

        if direction == 'interpolate':
            result = self.propagate(edge_index, x=x, mode='interpolate', parameters=parameters)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'fp':
            # Compute raw diffusiophoretic velocities (same as base PDE_D)
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)

            # Apply density-dependent modulation to particle velocities
            # In fp pass, x = x_particle_field = [mesh_nodes; particles]
            # Result has entries for all nodes; particles start at index n_nodes
            # local_density was computed for particle indices 0..n_particles-1
            if self.local_density is not None:
                n_total = x.size(0)
                n_particles = self.local_density.size(0)
                n_nodes = n_total - n_particles

                # Compute Hill function modulation for each particle
                ratio = self.local_density / self.rho_0
                modulation = 1.0 / (1.0 + ratio ** self.hill_n)  # shape [n_particles]

                # Apply modulation to particle portion of result
                # Mesh node entries (0..n_nodes-1) are unaffected
                mod_full = torch.ones(n_total, 1, device=x.device)
                mod_full[n_nodes:, 0] = modulation
                result = result * mod_full

            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            # First: compute local density for each particle (count neighbors within sensing radius)
            # This is stored and used by the 'fp' pass to modulate mobility
            self._compute_local_density(x, edge_index)

            # Then: compute standard pp forces
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """
        Count the number of particle neighbors within sensing_radius for each particle.
        Uses the pp edge_index which connects nearby particles.
        Stores result in self.local_density for use by fp pass.
        """
        n_particles = x.size(0)
        # Count neighbors per node from the edge index
        # edge_index[1] = target nodes, so count how many times each target appears
        target_nodes = edge_index[1]

        # Get positions to filter by actual distance
        pos_i = x[edge_index[1], 1:self.dimension+1]
        pos_j = x[edge_index[0], 1:self.dimension+1]
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))

        # Count neighbors within sensing radius
        within_radius = dist < self.sensing_radius
        counts = torch.zeros(n_particles, device=x.device)
        counts.scatter_add_(0, target_nodes[within_radius],
                           torch.ones(within_radius.sum(), device=x.device))

        self.local_density = counts

    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode. Same as base PDE_D.
        Density-dependent modulation is applied post-aggregation in forward().
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Same as base PDE_D — field interpolation is unaffected by density
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]
            weight = torch.exp(-dist / 0.01).unsqueeze(1)
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # Standard diffusiophoretic velocity computation (same as base PDE_D)
            # Density modulation is applied at forward() level after aggregation
            fields_i = x_i[:, 6:8]  # Particle fields [C1, C2]
            fields_j = x_j[:, 6:8]  # Mesh fields [C1, C2]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            # Smoothing kernel (same as base PDE_D)
            kernel = torch.exp(-dist / 0.05)

            # Direction vector
            dir_norm = d_pos / dist_safe.unsqueeze(1)

            # Gradient estimation (same as base PDE_D)
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            # Get mobility coefficients (per-type or global)
            if parameters_i is not None:
                M1 = parameters_i[:, 0:1]
                M2 = parameters_i[:, 1:2]
            else:
                M1 = self.M1
                M2 = self.M2

            # Raw diffusiophoretic velocity (density modulation applied in forward())
            velocities = (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            return velocities

        elif mode == 'pf':
            # Particle -> Field: same as base PDE_D
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            if parameters_i is not None:
                consumption = parameters_i[:, 2]
                production = parameters_i[:, 3]
            else:
                consumption = self.consumption_rate
                production = self.production_rate

            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle -> Particle: same as base PDE_D
            if parameters_i is not None:
                p1 = parameters_i[:, 4]
                p2 = parameters_i[:, 5]
                p3 = parameters_i[:, 6]
                p4 = parameters_i[:, 7]

                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))

                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)
            else:
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
        Process aggregated messages — same as base PDE_D.
        """
        if mode == 'interpolate':
            C1_weighted = aggr_out[:, 0:1]
            C2_weighted = aggr_out[:, 1:2]
            weight_sum = aggr_out[:, 2:3]
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            return torch.cat([C1_weighted / weight_sum, C2_weighted / weight_sum], dim=1)
        else:
            return aggr_out
