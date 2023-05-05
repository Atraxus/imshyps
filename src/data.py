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

def read_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config