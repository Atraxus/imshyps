import os
import matplotlib.pyplot as plt
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

from network import TrainData, TestNet
from data import read_config, create_hp_list, get_range, create_hp_combinations


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
    val_loss, val_acc = network.train(train_data)
    return val_loss, val_acc


def pretty_print(parameter: str, results: list):
    print("Results for " + parameter + ":")
    for result in results:
        print(
            "\t"
            + parameter.capitalize()
            + ": "
            + str(result[0])
            + ", loss: "
            + str(result[1])
            + ", accuracy: "
            + str(result[2])
        )


def plot_results(parameter: str, results: list):
    plt.plot(
        [result[0] for result in results],
        [result[1] for result in results],
        label="loss",
    )
    plt.plot(
        [result[0] for result in results],
        [result[2] for result in results],
        label="accuracy",
    )
    plt.xlabel(parameter)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_parameter(param_idx: int, parameter: list, results: pd.DataFrame):
    importance_score = 0

    losses = []
    accuracies = []
    for value in parameter:
        filtered_results = results[results.iloc[:, param_idx] == value]
        avg_loss = filtered_results["loss"].mean()
        avg_acc = filtered_results["accuracy"].mean()
        losses.append(avg_loss)
        accuracies.append(avg_acc)

    variance_loss = pd.Series(losses).var()
    variance_acc = pd.Series(accuracies).var()
    importance_score = variance_loss + variance_acc
    return importance_score

def analyze_results(hyperparameters: list, results: pd.DataFrame):
    # This function analyzes the importance of the hyperparameters
    importance_scores = []
    for i, parameter in enumerate(hyperparameters):
        importance_scores.append(analyze_parameter(i, parameter, results))
    return importance_scores


def main():
    config = read_config("configs/default.json")
    hyperparameters = create_hp_list(config)
    hp_permutations = create_hp_combinations(hyperparameters)
    train_data = TrainData()

    # Iterate over all hyperparameter permutations
    # The following dataframe will contain all permutations and the corresponding loss and accuracy for easier analysis
    results = []
    for permutation in hp_permutations:
        lrate, bsize, afun = permutation
        val_loss, val_acc = test_params(lrate, bsize, afun, train_data)
        results.append([lrate, bsize, afun, val_loss, val_acc])
    
    results = pd.DataFrame(results, columns=["learning_rate", "batch_size", "activation_function", "loss", "accuracy"])
    importance_scores = analyze_results(hyperparameters, results)
    
    print("Learning rate importance score: " + str(importance_scores[0]))
    print("Batch size importance score: " + str(importance_scores[1]))
    print("Activation function importance score: " + str(importance_scores[2]))

if __name__ == "__main__":
    main()
