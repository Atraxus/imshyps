from models import TrainData, Model
from hyperparameter import HyperParameter

import json
import numpy as np


# Activation functions in keras: relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
activation_functions = [
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential",
]


def is_activation_function(afun: str) -> bool:
    return afun in activation_functions


def get_samples(start: float, end: float, num_samples: int = 40) -> list:
    # Returns a list of floats from start to end with num_samples number of points
    return [start + i*(end-start)/(num_samples-1) for i in range(num_samples)]


# Returns samples for a given parameter definition
# Respects the type and removes duplicates
# E.g. if type is int and we have samples in [1,3] then we return 1,2,3
def get_param_samples(param_def):
    if "min" in param_def and "max" in param_def:
        if param_def["type"] == "float":
            return np.linspace(param_def["min"], param_def["max"], 40).tolist()
        elif param_def["type"] == "int":
            return np.unique(np.round(np.linspace(param_def["min"], param_def["max"], 40))).astype(int).tolist()
    elif "samples" in param_def:
        return param_def["samples"]
    else:
        return None


class ParamHandler:
    model: Model
    model_class: type
    MODEL_HPARAMS = []
    params: list
    train_data: TrainData
    metrics: list

    def __init__(self, model_class: type, model_hparams: list, config_path: str):
        self.model = None
        self.model_class = model_class
        self.MODEL_HPARAMS = model_hparams
        self.params_from_config(config_path)
        self.train_data = TrainData()

    def params_from_config(self, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        self.metrics = config["metrics"]
        cfg_params = config["params"]
        self.params = []
        for hp in self.MODEL_HPARAMS:
            if hp not in cfg_params:
                raise ValueError("Invalid config file")

            param_cfg = cfg_params[hp]
            samples = get_param_samples(param_cfg)
            default = param_cfg["default"]

            param = HyperParameter(hp, samples, default)
            self.params.append(param)

    def total_num_samples(self):
        return np.sum([len(param.samples) for param in self.params])

    def run(self):
        defaults = [(param.name, param.default) for param in self.params]
        results = []
        for i, param in enumerate(self.params):
            for value in param:
                param_dict = dict(defaults)
                param_dict[param.name] = value
                model = self.model_class(param_dict, self.metrics)
                print(
                    f"Running model with parameters {param_dict}")
                result = model.evaluate(self.train_data)
                results.append((result, param_dict))
        return results
