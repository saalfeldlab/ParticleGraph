from typing import Optional, Literal, Annotated, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ParticleGraph.config_manager import ConfigManager

# from ParticleGraph import (
#     GraphModel,
# )


# Sub-config schemas for ParticleGraph

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    params: list[list[float]]
    min_radius: Annotated[float, Field(ge=0)] = 0
    max_radius: Annotated[float, Field(gt=0)]
    diffusion_coefficients: Optional[list[float]] = None
    n_particles: int = 1000
    n_particle_types: int = 5
    n_interactions: int = 5
    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    has_cell_division: bool = False
    n_frames: int = 1000
    sigma: float = 0.005
    delta_t: float = 1
    dpos_init: float = 0
    boundary: Literal['periodic', 'no'] = 'periodic'
    node_value_map: Optional[str] = None
    node_type_map: Optional[str] = None
    beta: Optional[float] = None
    start_frame: int = 0


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    prediction: Literal['first_derivative', '2nd_derivative'] = '2nd_derivative'
    input_size: int
    output_size: int
    hidden_dim: int
    n_mp_layers: int
    aggr_type: str
    mesh_aggr_type: str = 'add'
    embedding_dim: int = 2
    update_type: Literal['linear', 'none'] = 'none'
    n_layers_update: int = 3
    hidden_dim_update: int = 64

    # def get_instance(self, **kwargs):
    #     return GraphModel(**self.model_dump(), **kwargs)


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    colormap: str = 'tab10'
    arrow_length: int = 10


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    n_epochs: int = 20
    batch_size: int = 1
    small_init_batch_size: bool = True

    n_runs: int = 2
    clamp: float = 0
    pred_limit: float = 1.E+10
    sparsity: Literal['none', 'replace'] = 'none'

    fix_cluster_embedding: bool = False
    loss_weight: bool = False
    learning_rate_start: float = 0.001
    learning_rate_end: float = 0.0005
    learning_rate_embedding_start: float = 0.001
    learning_rate_embedding_end: float = 0.001

    noise_level: float = 0
    data_augmentation: bool = True
    cluster_method: Literal['kmeans_auto', 'distance_plot', 'distance_embedding', 'distance_both'] = 'distance_plot'
    
    device: Annotated[str, Field(pattern=r'^(auto|cpu|cuda:\d+)$')] = 'auto'


# Main config schema for ParticleGraph

class ParticleGraphConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = 'ParticleGraph'
    dataset: str
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    plotting: PlottingConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, 'r') as file:
            raw_config = yaml.safe_load(file)
        return ParticleGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == '__main__':

    config_file = '../../config/arbitrary_3.yaml' # Insert path to config file
    config = ParticleGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print('Successfully loaded config file. Model description:', config.description)
    