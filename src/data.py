# File for reading json config files

# Example config file:
# {
#     "name": "default",
#     "learning_rate": {
#         "min": 0.0001,
#         "max": 0.001,
#         "step": 0.0001
#     },
#     "batch_size": {
#         "min": 32,
#         "max": 128,
#         "step": 32
#     },
#     "activation_functions": [
#         "relu",
#         "sigmoid",
#         "tanh"
#     ]
# }


import json
import pandas as pd
import itertools

def get_range(start: float, end: float, step: float) -> list:
    # Returns a list of floats from start to end with step size step
    return [start + i*step for i in range(int((end - start) // step) + 1)]

def read_config(path: str):
    # Reads a json config file and returns a dict
    with open(path, "r") as f:
        config = json.load(f)
    return config

def create_hp_list(config: dict):
    # Creates a list of all hyperparameters stripped of their names
    hyperparameters = []
    for key, value in config.items():
        if key == "name":
            continue
        if isinstance(value, dict):
            hyperparameters.append(get_range(value["min"], value["max"], value["step"]))
        elif isinstance(value, list):
            hyperparameters.append(value)
        else:
            raise ValueError("Invalid value type in config file.")
    return hyperparameters

def create_hp_combinations(hyperparameters: list):
    # Creates all possible permutations of the hyperparameters
    return [list(p) for p in itertools.product(*hyperparameters)]
