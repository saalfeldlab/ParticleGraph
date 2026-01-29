import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_ActiveMatter(pyg.nn.MessagePassing):
    """
    Active matter particle model combining self-propulsion with diffusiophoresis.

    Implements Vicsek-style self-propelled particles coupled to reaction-diffusion
    fields. Particles have intrinsic motility (self-propulsion speed v0) and
    align their heading with nearby neighbors, while also responding to
    concentration field gradients (chemotaxis/diffusiophoresis).

    Literature:
    - Vicsek et al. (1995) Physical Review Letters 75:1226-1229
      "Novel type of phase transition in a system of self-driven particles"
    - Cates & Tailleur (2015) Annual Review of Condensed Matter Physics 6:219-244
      "Motility-Induced Phase Separation"
    - Liebchen & Löwen (2018) Accounts of Chemical Research 51:2982-2990
      "Synthetic Chemotaxis and Collective Behavior in Active Matter"

    Physics:
    In standard diffusiophoresis: v = M * ∇C (passive response to gradients)
    In active matter:
      v = v0 * heading + M * ∇C + alignment_contribution + noise
    where:
      - heading = normalized velocity direction (persistent motility)
      - v0 = self-propulsion speed (intrinsic motility)
      - alignment_contribution = tendency to align with neighbor velocities (Vicsek)
      - gradient_bias modulates how strongly ∇C rotates heading vs. directly drives motion

    Key differences from linear PDE_D:
    1. Self-propulsion: Particles move even without gradients
    2. Velocity alignment: Neighbors influence heading direction (polar order)
    3. Motility-induced phase separation (MIPS): Self-propulsion + density-dependent
       slowdown can create clusters without attractive interactions
    4. New collective states: flocking bands, vortices, polar lanes

    Per-type params layout: [v0, alignment, gradient_bias, noise_amp, ar_p1, ar_p2, ar_p3, ar_p4]
      - v0: Self-propulsion speed (intrinsic motility)
      - alignment: Velocity alignment strength (Vicsek coupling)
      - gradient_bias: How strongly field gradients influence heading (0=pure active, 1=pure diffusiophoresis)
      - noise_amp: Angular noise amplitude (rotational diffusion)
      - ar_p1-4: Attraction-repulsion parameters (same as base PDE_D)
    """

    PARAMS_DOC = {
        "model_name": "ActiveMatter",
        "literature": "Vicsek et al. (1995) PRL 75:1226; Cates & Tailleur (2015) ARCMP 6:219",
        "description": "Self-propelled particles with Vicsek alignment + diffusiophoretic coupling",
        "equations": {
            "velocity": "v = v0 * heading + gradient_bias * (M1*∇C1 + M2*∇C2) + alignment",
            "alignment": "heading_new = normalize(Σ_neighbors v_j / |Σ v_j|) * alignment_strength",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2σ²)) - p3*exp(-d^(2p4)/(2σ²))) * dir + alignment"
        },
        "params": [
            {"index": 0, "name": "v0", "description": "Self-propulsion speed", "typical_range": [0.1, 2.0]},
            {"index": 1, "name": "alignment", "description": "Vicsek alignment strength", "typical_range": [0.0, 1.0]},
            {"index": 2, "name": "gradient_bias", "description": "Gradient response strength (like consumption in base)", "typical_range": [0.0, 1.0]},
            {"index": 3, "name": "noise_amp", "description": "Angular noise (rotational diffusion)", "typical_range": [0.0, 0.5]},
            {"index": 4, "name": "ar_p1", "description": "Attraction strength", "typical_range": [0.5, 3.0]},
            {"index": 5, "name": "ar_p2", "description": "Attraction exponent", "typical_range": [0.5, 2.0]},
            {"index": 6, "name": "ar_p3", "description": "Repulsion strength", "typical_range": [0.5, 3.0]},
            {"index": 7, "name": "ar_p4", "description": "Repulsion exponent", "typical_range": [0.5, 2.0]}
        ]
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_ActiveMatter, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Global parameters from mesh
        # Diffusiophoretic mobility for gradient response
        # Note: p[0,5] is shared with mesh model (chi/time_scale) so we use
        # hardcoded defaults for M1/M2. The gradient_bias per-type parameter
        # scales the overall gradient response strength.
        self.M1 = torch.tensor(-4.0, device=p.device)  # Standard M1 for C1 gradients
        self.M2 = torch.tensor(4.0, device=p.device)   # Standard M2 for C2 gradients

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Peclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters (same as base PDE_D)
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Active matter parameters (global defaults, overridden by per-type params)
        # v0: self-propulsion speed
        self.v0_default = 0.5
        # alignment_strength: Vicsek coupling
        self.alignment_default = 0.3
        # gradient_bias: how strongly gradients contribute (vs self-propulsion)
        self.gradient_bias_default = 0.5
        # noise_amp: rotational noise
        self.noise_amp_default = 0.1

        # Report configuration
        print(f"initialized PDE_D_ActiveMatter with parameters:")
        print(f"mobility: M₁={self.M1.item()}, M₂={self.M2.item()}")
        print(f"active matter: v0={self.v0_default}, alignment={self.alignment_default}")
        print(f"gradient_bias={self.gradient_bias_default}, noise_amp={self.noise_amp_default}")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle→Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [v0, alignment, gradient_bias, noise_amp, ar_p1, ar_p2, ar_p3, ar_p4]")

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
                    f"PDE_D_ActiveMatter: particle_params has {n_param_rows} rows but found "
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
            # Field-to-particle: diffusiophoresis + self-propulsion
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)

            # Add self-propulsion component based on current velocity
            vel = x[:, 3:3+self.dimension]  # Current velocity [vx, vy]
            speed = torch.sqrt(torch.sum(vel**2, dim=1, keepdim=True))
            speed_safe = torch.clamp(speed, min=1e-6)
            heading = vel / speed_safe  # Normalized heading direction

            # Get v0 per particle type
            if parameters is not None:
                v0 = parameters[:, 0:1]
                noise_amp = parameters[:, 3:4]
            else:
                v0 = self.v0_default
                noise_amp = self.noise_amp_default

            # Self-propulsion: constant speed in heading direction
            self_propulsion = v0 * heading

            # Angular noise: rotate heading by random angle
            # This creates rotational diffusion (persistent random walk)
            if isinstance(noise_amp, float):
                if noise_amp > 0:
                    angle_noise = noise_amp * torch.randn(x.size(0), 1, device=x.device)
                    cos_a = torch.cos(angle_noise)
                    sin_a = torch.sin(angle_noise)
                    noisy_heading = torch.zeros_like(heading)
                    noisy_heading[:, 0:1] = heading[:, 0:1] * cos_a - heading[:, 1:2] * sin_a
                    noisy_heading[:, 1:2] = heading[:, 0:1] * sin_a + heading[:, 1:2] * cos_a
                    self_propulsion = v0 * noisy_heading
            else:
                angle_noise = noise_amp * torch.randn(x.size(0), 1, device=x.device)
                cos_a = torch.cos(angle_noise)
                sin_a = torch.sin(angle_noise)
                noisy_heading = torch.zeros_like(heading)
                noisy_heading[:, 0:1] = heading[:, 0:1] * cos_a - heading[:, 1:2] * sin_a
                noisy_heading[:, 1:2] = heading[:, 0:1] * sin_a + heading[:, 1:2] * cos_a
                self_propulsion = v0 * noisy_heading

            # Combine: gradient response + self-propulsion
            # The gradient response from propagate is already scaled by gradient_bias
            result = result + self_propulsion

            # Zero out for out-of-box particles
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()

            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            # Particle-particle: attraction-repulsion + velocity alignment
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode.

        Key differences from base PDE_D:
        - 'fp' mode: gradient response scaled by gradient_bias parameter
        - 'pp' mode: adds Vicsek-style velocity alignment to attraction-repulsion
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Same as base PDE_D — field interpolation
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]
            weight = torch.exp(-dist / 0.01).unsqueeze(1)
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # Diffusiophoretic gradient response (scaled by gradient_bias)
            fields_i = x_i[:, 6:8]
            fields_j = x_j[:, 6:8]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            kernel = torch.exp(-dist / 0.05)
            dir_norm = d_pos / dist_safe.unsqueeze(1)
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            # Get gradient_bias (how much gradients contribute)
            if parameters_i is not None:
                gradient_bias = parameters_i[:, 2:3]
            else:
                gradient_bias = self.gradient_bias_default

            # Use global mobility (M1, M2) scaled by gradient_bias
            M1 = self.M1
            M2 = self.M2

            # Gradient-driven velocity (scaled by gradient_bias)
            velocities = gradient_bias * (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            return velocities

        elif mode == 'pf':
            # Particle → Field: consumption/production (uses gradient_bias and noise_amp
            # as consumption and production respectively)
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            # For pf mode, use consumption_rate and production_rate from mesh params
            # (NOT from per-type params which have different meaning in ActiveMatter)
            consumption = self.consumption_rate
            production = self.production_rate

            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle-particle: attraction-repulsion + Vicsek velocity alignment

            # Get velocities for alignment
            vel_i = x_i[:, 3:3+self.dimension]
            vel_j = x_j[:, 3:3+self.dimension]

            if parameters_i is not None:
                alignment_strength = parameters_i[:, 1:2]
                p1 = parameters_i[:, 4]
                p2 = parameters_i[:, 5]
                p3 = parameters_i[:, 6]
                p4 = parameters_i[:, 7]

                # Attraction-repulsion forces (same as base PDE_D)
                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))
                ar_forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)

                # Vicsek alignment: tendency to match neighbor velocity direction
                # Weight alignment by proximity (closer neighbors have stronger effect)
                alignment_weight = torch.exp(-dist / 0.05)
                # Velocity difference -> alignment force
                alignment_force = alignment_strength * alignment_weight.unsqueeze(1) * (vel_j - vel_i)

                forces = ar_forces + alignment_force
            else:
                # Fallback: simple repulsion + alignment
                forces = torch.zeros_like(pos_i)
                in_range = dist < self.repulsion_range
                if in_range.any():
                    dir_norm = d_pos / dist_safe.unsqueeze(1)
                    repulsion_mag = self.repulsion_strength * torch.exp(
                        -5.0 * dist[in_range] / self.repulsion_range
                    )
                    forces[in_range] = -dir_norm[in_range] * repulsion_mag.unsqueeze(1)

                # Add alignment (Vicsek)
                alignment_weight = torch.exp(-dist / 0.05)
                alignment_force = self.alignment_default * alignment_weight.unsqueeze(1) * (vel_j - vel_i)
                forces = forces + alignment_force

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
