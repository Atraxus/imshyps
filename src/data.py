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


import itertools
import random
import json
import pandas as pd
from paramhandler import get_param_samples


def get_samples(start: float, end: float, num_samples: int = 40) -> list:
    # Returns a list of floats from start to end with num_samples number of points
    return [start + i*(end-start)/(num_samples-1) for i in range(num_samples)]


def read_config(path: str):
    # Reads a json config file and returns a list of hyperparameters
    with open(path, "r") as f:
        config = json.load(f)
    hyperparameters = []


def generate_random_results(config_path: str = "configs/mlp.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    cfg_params = config["params"]
    default_params = {name: details["default"]
                      for name, details in cfg_params.items()}

    results = []
    seen_hyperparameters = set()  # This set keeps track of already seen hyperparameters
    for param_name, param_def in cfg_params.items():
        param_samples = get_param_samples(param_def)
        for sample in param_samples:
            params = default_params.copy()
            params[param_name] = sample

            # Convert params dictionary to a tuple of sorted items
            params_tuple = tuple(sorted(params.items()))
            if params_tuple in seen_hyperparameters:
                # If this set of hyperparameters has been seen before, skip it
                continue

            seen_hyperparameters.add(params_tuple)

            # Generating a random accuracy for the dummy result
            result = random.uniform(0, 1)
            results.append((result, params))

    return results
