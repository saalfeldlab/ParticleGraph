import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_Anisotropic(pyg.nn.MessagePassing):
    """
    Anisotropic diffusiophoresis particle model.

    Particles respond differently to concentration gradients in x vs y direction,
    breaking the isotropy of standard diffusiophoresis. This models contact guidance
    and oriented cell motility on substrates with aligned extracellular matrix fibers.

    Literature:
    - Tranquillo & Murray (1992) J Math Biol 31:583-600
      "Continuum model of fibroblast-driven wound contraction: contact guidance"
    - Dickinson & Tranquillo (1993) Ann Biomed Eng 21:679-691
      "A stochastic model for adhesion-mediated cell random motility and haptotaxis"
    - Painter (2009) J Math Biol 58:511-543
      "Modelling cell migration strategies in the extracellular matrix"

    Physics:
    In standard diffusiophoresis: v = M * nabla(C) (isotropic)
    In anisotropic:              vx = alpha * M * dC/dx,  vy = M * dC/dy
    where alpha = anisotropy ratio (0 < alpha < 1 means weaker x-response,
                                     alpha > 1 means stronger x-response)

    Key differences from linear PDE_D:
    1. Directional selectivity: Particles preferentially move along one axis
    2. Symmetry breaking: Hexagonal spots can elongate into ellipses or stripes
    3. Orientation control: Combined with Turing patterns, selects stripe orientation

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
      Same as base PDE_D for compatibility.

    Anisotropy is controlled by p[2, 4] (alpha parameter):
      alpha < 1: weaker response in x, stronger in y -> vertical elongation
      alpha = 1: isotropic (same as base PDE_D)
      alpha > 1: stronger response in x, weaker in y -> horizontal elongation
    """

    PARAMS_DOC = {
        "model_name": "Anisotropic",
        "literature": "Tranquillo & Murray (1992) J Math Biol 31:583-600",
        "description": "Anisotropic diffusiophoretic mobility with directional selectivity",
        "equations": {
            "field_to_particle": "vx = alpha * (M1 * dC1/dx + M2 * dC2/dx), vy = (M1 * dC1/dy + M2 * dC2/dy)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2s^2)) - p3*exp(-d^(2p4)/(2s^2))) * dir"
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
        "anisotropy_params": {
            "alpha": "Anisotropy ratio (x-mobility / y-mobility). From p[2,4]. alpha<1: y-preferred, alpha=1: isotropic, alpha>1: x-preferred. Default=0.25",
            "note": "Only affects the field-to-particle (fp) interaction. Particle-field and particle-particle remain isotropic."
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_Anisotropic, self).__init__(aggr=aggr_type)

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

        # Anisotropy ratio alpha from p[2, 4]
        # alpha < 1: weaker x-response (particles prefer moving in y)
        # alpha = 1: isotropic (recovers standard PDE_D)
        # alpha > 1: stronger x-response (particles prefer moving in x)
        if p.shape[1] > 4:
            self.alpha = p[2, 4]
        else:
            self.alpha = torch.tensor(0.25, device=p.device)

        # Report configuration
        alpha_val = self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha
        print(f"initialized PDE_D_Anisotropic with parameters:")
        print(f"mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"anisotropy alpha={alpha_val:.3f} (x/y mobility ratio)")
        if alpha_val < 1:
            print(f"  -> y-preferred motion (vertical elongation expected)")
        elif alpha_val > 1:
            print(f"  -> x-preferred motion (horizontal elongation expected)")
        else:
            print(f"  -> isotropic (same as base PDE_D)")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle->Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")

    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction.
        Same interface as base PDE_D for compatibility.
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
                    f"PDE_D_Anisotropic: particle_params has {n_param_rows} rows but found "
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
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode.

        Key difference from base PDE_D: In 'fp' mode, the x-component of the
        diffusiophoretic velocity is scaled by alpha, creating anisotropic response.
        This breaks radial symmetry and can elongate clusters or select stripe orientation.
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Same as base PDE_D
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]
            weight = torch.exp(-dist / 0.01).unsqueeze(1)
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # ANISOTROPIC DIFFUSIOPHORESIS
            # Instead of v = M * grad_C * dir (isotropic),
            # we compute: vx = alpha * M * grad_C * dir_x, vy = M * grad_C * dir_y
            fields_i = x_i[:, 6:8]
            fields_j = x_j[:, 6:8]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            # Smoothing kernel (same as base PDE_D)
            kernel = torch.exp(-dist / 0.05)

            # Direction vector (2D: [dx, dy])
            dir_norm = d_pos / dist_safe.unsqueeze(1)

            # Gradient estimation
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

            # Compute isotropic velocity components first
            velocity_scalar = M1 * grad_C1 + M2 * grad_C2  # [N, 1]

            # Apply anisotropy: scale x-component by alpha, keep y-component unchanged
            # This creates directional preference:
            # alpha < 1: weaker x-response -> vertical stripe/elongation selection
            # alpha > 1: stronger x-response -> horizontal stripe/elongation selection
            aniso_scale = torch.ones_like(dir_norm)  # [N, 2]
            aniso_scale[:, 0] = self.alpha  # Scale x-component
            # y-component stays 1.0 (reference direction)

            velocities = velocity_scalar * dir_norm * aniso_scale

            return velocities

        elif mode == 'pf':
            # Particle -> Field: same as base PDE_D (isotropic)
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
            # Particle -> Particle: same as base PDE_D (isotropic)
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
