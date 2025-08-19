from typing import Optional, List
import yaml
from pydantic import BaseModel, ConfigDict


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.18


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    T: float = 1000.0
    dt: float = 0.1
    v_init: float = -1.0
    w_init: float = 1.0
    pulse_interval: float = 80.0
    pulse_duration: float = 1.0
    pulse_amplitude: float = 0.8
    noise_level: float = 1E-3


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    siren_in_features: int = 1
    siren_out_features: int = 1
    siren_hidden_features: int = 128
    siren_hidden_layers: int = 3
    siren_outermost_linear: bool = True
    mlp0_input_size: int = 3
    mlp0_output_size: int = 1
    mlp0_nlayers: int = 5
    mlp0_hidden_size: int = 128
    mlp1_input_size: int = 2
    mlp1_output_size: int = 1
    mlp1_nlayers: int = 2
    mlp1_hidden_size: int = 4


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "auto"
    n_iter: int = 5000
    batch_ratio: float = 0.2
    test_runs: int = 5
    learning_rate: float = 1e-4
    l1_lambda: float = 1.0e-3
    weight_decay: float = 1e-6
    recursive_loop: int = 3
    recursive_weight: list = [0.33, 0.66, 1.0, 1.33, 1.66]  # Weights for recursive updates
    n_init_steps: int = 1000
    use_siren_init: bool = False
    lambda_jac: float = 0
    lambda_ratio: float = 0
    lambda_amp: float = 0

    # Jacobian sensitivity thresholds
    tau_vv: float = 0.10  # Threshold for ∂dv/∂v sensitivity
    tau_vw: float = 0.10  # Threshold for ∂dv/∂w sensitivity
    tau_wv: float = 0.10  # Threshold for ∂dw/∂v sensitivity
    tau_ww: float = 0.05  # Threshold for ∂dw/∂w sensitivity (weaker)


class FitzhughNagumoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Optional[str] = "FitzHugh-Nagumo Neural Dynamics"
    system: SystemConfig
    simulation: SimulationConfig
    model: ModelConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return FitzhughNagumoConfig(**raw_config)

    def to_yaml(self, file_name: str):
        with open(file_name, "w") as file:
            yaml.dump(self.model_dump(), file, default_flow_style=False, sort_keys=False, indent=2)