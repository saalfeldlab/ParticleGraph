import json
import yaml
from cerberus import Validator
import astropy.units as u
import math
import re
from abc import ABC
import os
from importlib.resources import path

def create_config_manager(config_type):
    if config_type == 'simulation':
        return ConfigManagerSimulation()
    elif config_type == 'experiment':
        return ConfigManagerExperiment()
    else:
        raise ValueError('Invalid config type!')

def astropy_constructor(loader, node):
    value = loader.construct_scalar(node)
    if value == '':
        return ''
    else:
        return u.Quantity(value)

def math_constructor(loader, node):
    expression = loader.construct_scalar(node)
    # Use eval safely by restricting globals and providing only the math module functions
    return eval(expression, {"__builtins__": None}, math.__dict__)

def register_yaml_constructors():
    yaml.add_constructor('!astropy', astropy_constructor)
    yaml.add_constructor('!math', math_constructor)
    yaml.SafeLoader.add_constructor('!astropy', astropy_constructor)
    yaml.SafeLoader.add_constructor('!math', math_constructor)

class CustomValidator(Validator):
    def _validate_type_astropy(self, value):
        if not isinstance(value, u.Quantity):
            return False # validation failure
        return True # successful validation


class ConfigManager(ABC):
    def __init__(self, config_schema=None):
        register_yaml_constructors()
        self.config_schema = ConfigManager.load_config_schema(config_schema)

    def get_config(self):
        return self.config

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def load_and_validate_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        v = CustomValidator(self.config_schema)
        v.allow_unknown = False
        config = v.normalized(config)
        if not v.validate(config):
            raise ValueError(f"Invalid configuration: {v.errors}")

        return config
    
    @staticmethod
    def load_config_schema(schema_file):

        path = os.path.join(os.path.dirname(__file__), schema_file)
        if 'modules' in path:
            path = path.replace('modules', '')

        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def load_config(config_file):
        register_yaml_constructors()
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
        
class ConfigManagerSimulation(ConfigManager):
    def __init__(self):
        with path('ParticleGraph.config_schemas', 'config_schema_simulation.yaml') as config_path:
            super().__init__(str(config_path))

class ConfigManagerExperiment(ConfigManager):
    def __init__(self):
        with path('ParticleGraph.config_schemas', 'config_schema_experiment.yaml') as config_path:
            super().__init__(str(config_path))
