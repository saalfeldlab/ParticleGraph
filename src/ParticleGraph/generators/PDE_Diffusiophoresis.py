import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis(pyg.nn.MessagePassing):
    """
    Reaction-diffusion model for diffusiophoresis assembly simulation.
    Implements Brusselator model for two concentration fields (C₁, C₂).
    
    Inputs
    ----------
    data : a torch_geometric.data object
    
    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields C₁ and C₂
    """
    
    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis, self).__init__(aggr='add')  # "add" aggregation
        
        self.bc_dpos = bc_dpos
        
        # Initialize parameters directly from p tensor
        # C1 parameters
        self.D1 = p[0, 0]       # Diffusion coefficient for C₁
        self.Da_c = p[0, 1]     # Damköhler number
        self.A = p[0, 2]        # Brusselator parameter A
        self.B = p[0, 3]        # Brusselator parameter B
        
        # Add morphological parameter μ if available, otherwise use default
        if p[0].size(0) > 4:
            self.mu = p[0, 4]   # Morphological parameter
        else:
            self.mu = torch.tensor(0.04, device=p.device)  # Default for hexagonal patterns
        
        # C2 parameters
        self.D2 = p[1, 0]       # Diffusion coefficient for C₂
        
        # Store coefficient for later use
        self.coeff = p
        
        # Print initialized parameters for verification
        print(f"initialized PDE_Diffusiophoresis with parameters:")
        print(f"C₁: D={self.D1.item():.3f}, C₂: D={self.D2.item():.3f}, Da_c={self.Da_c.item():.3f}, A={self.A.item():.3f}, B={self.B.item():.3f}, μ={self.mu.item():.3f}")
    
    def forward(self, data):
        """
        Update the concentration fields using the Brusselator reaction-diffusion model
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values are at indices 6 and 7 (C₁ and C₂)
            - edge_index: Connectivity [2, n_edges]
            - edge_attr: Edge attributes (Laplacian coefficients) [n_edges]
            
        Returns
        -------
        d_C : torch.Tensor
            The derivatives of C₁ and C₂ [n_nodes, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Extract field values
        C1 = x[:, 6:7]  # C₁ at index 6
        C2 = x[:, 7:8]  # C₂ at index 7
        
        # Compute Laplacians for both fields
        laplacian_C1 = self.propagate(edge_index, u=C1, edge_attr=edge_attr)
        laplacian_C2 = self.propagate(edge_index, u=C2, edge_attr=edge_attr)
        
        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_C1
        self.laplacian_C2 = laplacian_C2
        
        # Apply diffusion with D coefficients
        diff_C1 = self.D1 * laplacian_C1
        diff_C2 = self.D2 * laplacian_C2
        
        # Compute reaction terms (Brusselator model)
        R1 = self.Da_c * (self.A - (self.B+1)*C1 + C1*C1*C2)
        R2 = self.Da_c * (self.B*C1 - C1*C1*C2)
        
        # Combine diffusion and reaction terms
        # dC1 = diff_C1 + R1
        # dC2 = diff_C2 + R2

        damping = damping = 0.005
        dC1 = diff_C1 + R1 - damping * (C1 - self.A)
        dC2 = diff_C2 + R2 - damping * (C2 - self.B/self.A)
        
        # Combine derivatives
        d_C = torch.cat([dC1, dC2], dim=1)
        
        return d_C
    

    def message(self, u_j, edge_attr):

        
        L = edge_attr[:, None] * u_j
        return L
    
    # def message(self, u_i, u_j, edge_attr):

    #     L = edge_attr[:, None] * (u_j - u_i)
    #     return L