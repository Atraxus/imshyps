import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

from network import TestNetwork
from data import read_config


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


def get_range(start: float, end: float, step: float) -> list:
    array = []
    while start <= end + step:
        array.append(start)
        start += step
    return array

def test_params(lrate: float, bsize: int, afun: str) -> tuple:
    if not is_activation_function(afun):
        raise ValueError("Invalid activation function")
    network = TestNetwork(lrate, bsize, afun)
    network.load_data()
    val_loss, val_acc = network.train()
    return val_loss, val_acc

def pretty_print(parameter: str, results: list):
    print("Results for " + parameter + ":")
    for result in results:
        print("\t" + parameter.capitalize() + ": " + str(result[0]) + ", loss: " + str(result[1]) + ", accuracy: " + str(result[2]))

def plot_results(parameter: str, results: list):
    plt.plot([result[0] for result in results], [result[1] for result in results], label="loss")
    plt.plot([result[0] for result in results], [result[2] for result in results], label="accuracy")
    plt.xlabel(parameter)
    plt.legend()
    plt.show()

def main():
    config = read_config("configs/default.json")

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    activation_functions = config["activation_functions"]

    rates = get_range(learning_rate["min"], learning_rate["max"], learning_rate["step"])
    sizes = get_range(batch_size["min"], batch_size["max"], batch_size["step"])

    def_lrate = rates[round(len(rates) / 2)]
    def_bsize = sizes[round(len(sizes) / 2)]
    def_afunc = activation_functions[0]
    
    results = []
    for lrate in rates:
        print("\n\nTesting learning rate: " + str(lrate))
        result = test_params(lrate, def_bsize, def_afunc)
        results.append((lrate, result[0], result[1]))
    pretty_print("learning rate", results)
    plot_results("learning rate", results)

    results = []
    for bsize in sizes:
        print("\n\nTesting batch size: " + str(bsize))
        result = test_params(def_lrate, bsize, def_afunc)
        results.append((bsize, result[0], result[1]))
    pretty_print("batch size", results)

    results = []
    for afunc in activation_functions:
        print("\n\nTesting activation function: " + afunc)
        result = test_params(def_lrate, def_bsize, afunc)
        results.append((afunc, result[0], result[1]))
    pretty_print("activation function", results)


if __name__ == "__main__":
    main()