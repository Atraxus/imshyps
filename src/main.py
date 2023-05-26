import os
import matplotlib.pyplot as plt
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

from network import TrainData, TestNet
from data import read_config, create_hp_list, get_samples


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


def test_params(lrate: float, bsize: int, afun: str, train_data: TrainData) -> tuple:
    if not is_activation_function(afun):
        raise ValueError("Invalid activation function")
    network = TestNet(lrate, bsize, afun)
    validation_loss = network.train(train_data)
    return validation_loss

def fanova(results: list):
    # This function applies the fANOVA algorithm to the results of a single hyperparameter and returns the importance score
    val_losses = []
    for result in results:
        val_losses.append(sum(result[3]) / len(result[3])) # Average validation loss over all runs

    variance = sum([(val_loss - sum(val_losses) / len(val_losses))**2 for val_loss in val_losses]) / len(val_losses)
    return variance


def analyze_results(hyperparameters: list, results: list):
    # This function analyzes the importance of the hyperparameters
    lrate_importance = fanova(results[0])
    bsize_importance = fanova(results[1])
    afun_importance =  fanova(results[2])

    print("Learning rate importance score: " + str(lrate_importance))
    print("Batch size importance score: " + str(bsize_importance))
    print("Activation function importance score: " + str(afun_importance))


def main():
    config = read_config("configs/default.json")
    hyperparameters = create_hp_list(config)
    train_data = TrainData()

    default_lrate = hyperparameters[0][len(hyperparameters[0]) // 2]
    default_bsize = hyperparameters[1][len(hyperparameters[1]) // 2]
    default_afun = hyperparameters[2][len(hyperparameters[2]) // 2]

    # Iterate over all hyperparameters
    results = []
    for i, hyperparameter in enumerate(hyperparameters):
        hp_results = []
        for value in hyperparameter:
            if i == 0: # learning rate
                lrate = value
                bsize = int(default_bsize)
                afun = default_afun
            elif i == 1: # batch size
                lrate = default_lrate
                bsize = int(value)
                afun = default_afun
            elif i == 2: # activation function
                lrate = default_lrate
                bsize = int(default_bsize)
                afun = value
            else:
                raise ValueError("Invalid hyperparameter index")
        
            validation_loss = test_params(lrate, bsize, afun, train_data)
            hp_results.append([lrate, bsize, afun, validation_loss])
        results.append(hp_results)
    
    analyze_results(hyperparameters, results)

if __name__ == "__main__":
    main()
