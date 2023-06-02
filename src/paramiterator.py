
from data import read_config, create_hp_list
from network import TrainData, TestNet

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


class ParamIterator():
    # param_list = []
    # default_params = []

    def __init__(self, config_path: str = "configs/default.json"):
        self.config = read_config(config_path)
        self.hyperparameters = create_hp_list(self.config)
        self.train_data = TrainData()

        self.default_lrate = self.hyperparameters[0][len(
            self.hyperparameters[0]) // 2]
        self.default_bsize = self.hyperparameters[1][len(
            self.hyperparameters[1]) // 2]
        self.default_afun = self.hyperparameters[2][len(
            self.hyperparameters[2]) // 2]

    # Iterate over all hyperparameters
    def iterate(self):
        results = []
        for i, hyperparameter in enumerate(self.hyperparameters):
            hp_results = []
            for value in hyperparameter:
                if i == 0:  # learning rate
                    lrate = value
                    bsize = int(self.default_bsize)
                    afun = self.default_afun
                elif i == 1:  # batch size
                    lrate = self.default_lrate
                    bsize = int(value)
                    afun = self.default_afun
                elif i == 2:  # activation function
                    lrate = self.default_lrate
                    bsize = int(self.default_bsize)
                    afun = value
                else:
                    raise ValueError("Invalid hyperparameter index")

                validation_loss = test_params(
                    lrate, bsize, afun, self.train_data)
                hp_results.append([lrate, bsize, afun, validation_loss])
            results.append(hp_results)

        return results
