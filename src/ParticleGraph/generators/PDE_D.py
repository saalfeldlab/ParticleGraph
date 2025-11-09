import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import *

class PDE_D(pyg.nn.MessagePassing):
    """
    Computes interactions between particles and fields in both directions.
    Can handle field→particle (diffusiophoresis motion) and particle→field (source/sink) interactions.
    
    Inputs
    ----------
    data : a torch_geometric.data object
    direction : str
        'fp' for field→particle, 'pf' for particle→field
    
    Returns
    -------
    For field→particle: particle velocities based on field gradients
    For particle→field: field updates based on particle effects
    """
    
    def __init__(self, aggr_type='mean', p=None, bc_dpos=None, dimension=2):
        super(PDE_D, self).__init__(aggr=aggr_type)  # "mean" aggregation
        
        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        
        # Mobility coefficients for diffusiophoresis
        self.M1 = p[0, 5]       # Mobility coefficient for C₁ (typically negative)
        self.M2 = p[1, 1]       # Mobility coefficient for C₂ (typically positive)
        
        # Particle effects on fields
        self.consumption_rate = p[2, 1] # Consumption rate of C₁
        self.production_rate = p[2, 2]  # Production rate of C₂
            
        # Influence radius - controls spatial extent of particle effects
        self.influence_radius = p[2, 3]
  
        # Print parameters for verification
        print(f"initialized PDE_D with parameters:")
        print(f"mobility: M₁={self.M1.item()}, M₂={self.M2.item()}")
        print(f"particle Effects: Consumption={self.consumption_rate.item()}, Production={self.production_rate.item()}, Radius={self.influence_radius.item()}")

    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data
        direction : str
            'fp' for field→particle, 'pf' for particle→field
            
        Returns
        -------
        torch.Tensor
            Interactions (velocities or field updates)
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
    
        fields = x[:, 6:8]

        # Propagate messages based on direction
        if direction == 'fp':
            # Field → Particle (diffusiophoresis)
            result = self.propagate(edge_index, x=x, fields=fields, mode='fp')
        else:
            # Particle → Field (consumption/production)
            result = self.propagate(edge_index, x=x, fields=fields, mode='pf')
            
        return result
    def message(self, edge_index_i, edge_index_j, x_i, x_j, fields_i, fields_j, mode):
        """
        Compute messages based on direction
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]
        
        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        
        if mode == 'fp':
            # Field → Particle: Calculate diffusiophoresis velocities
            # Avoid division by zero
            dist_safe = torch.clamp(dist, min=1e-6)
            
            # Normalized direction vectors
            dir_norm = d_pos / dist_safe.unsqueeze(1)
            
            # Gradient calculation (field value * direction / distance)
            grad_C1 = dir_norm * fields_j[:, 0:1]
            grad_C2 = dir_norm * fields_j[:, 1:2]
            
            # Apply mobility coefficients to get diffusiophoresis velocity
            return self.M1 * grad_C1 + self.M2 * grad_C2
        
        else:  # mode == 'pf'
            # Particle → Field: Calculate field updates
            # Gaussian influence based on distance and influence radius
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))
            
            # Create and return field updates directly
            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -self.consumption_rate * weights
            field_updates[:, 1] = self.production_rate * weights
            
            return field_updates
    
    def update(self, aggr_out):
        # Simple pass-through of aggregated messages
        return aggr_out