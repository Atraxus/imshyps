# File for reading json config files

# Example config file:
# {
#     "name": "default",
#     "learning_rate": {
#         "min": 0.0001,
#         "max": 0.001
#     },
#     "batch_size": {
#         "min": 32,
#         "max": 128
#     },
#     "activation_functions": [
#         "relu",
#         "sigmoid",
#         "tanh"
#     ]
# }


import json
import pandas as pd
from paramhandler import HyperParameter, hp_type


def get_samples(start: float, end: float, num_samples: int = 40) -> list:
    # Returns a list of floats from start to end with num_samples number of points
    return [start + i*(end-start)/(num_samples-1) for i in range(num_samples)]


def read_config(path: str):
    # Reads a json config file and returns a list of hyperparameters
    with open(path, "r") as f:
        config = json.load(f)
    hyperparameters = []
    
