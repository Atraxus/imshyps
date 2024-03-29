import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


# This needs to be here to avoid stdout spam from tensorflow
import tensorflow as tf  # trunk-ignore(ruff/E402)
import tensorflow_addons as tfa  # trunk-ignore(ruff/E402)
from tensorflow import keras  # trunk-ignore(ruff/E402)

tf.get_logger().setLevel("ERROR")


class TrainData:
    x_train: tf.Tensor
    y_train: tf.Tensor
    x_test: tf.Tensor
    y_test: tf.Tensor

    def __init__(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_test: tf.Tensor,
        y_test: tf.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class Model(ABC):
    # List of hyperparameters that the model needs. These will be read from
    # the config file. If a config file does not contain all of these, it is
    # invalid. Parameters that are not in this list will be ignored.
    MODEL_HPARAMS = []

    @abstractmethod
    # Initialize the model with the given hyperparameters. It is a dict with
    # key-value pairs. The default metric used for evaluation is accuracy.
    def __init__(self, hyperparameters: dict):
        pass

    @abstractmethod
    # Train the model and return the accuracy
    def load_data(test_size: float = 0.2):
        pass

    @abstractmethod
    # Train the model and return the accuracy
    def evaluate(self, train_data: TrainData):
        pass


class MLP(Model):
    model: keras.Sequential
    hyperparameters: dict
    MODEL_HPARAMS = [
        "learning_rate",
        "batch_size",
        "activation_function",
        "num_layers",
        "num_neurons",
    ]
    epochs: int

    def __init__(self, hyperparameters: dict, epochs: int):
        self.hyperparameters = hyperparameters
        self.epochs = epochs

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Flatten())
        for _ in range(self.hyperparameters["num_layers"]):
            self.model.add(
                keras.layers.Dense(
                    self.hyperparameters["num_neurons"],
                    activation=self.hyperparameters["activation_function"],
                )
            )
        self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"]
        )
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Get the TrainData object from the MNIST dataset
    def load_data(test_size: float = 0.2):
        fashion_mnist = keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return TrainData(x_train, y_train, x_test, y_test)

    # Train the model and return the accuracy
    def evaluate(self, train_data: TrainData):
        self.model.fit(
            train_data.x_train,
            train_data.y_train,
            epochs=self.epochs,
            batch_size=self.hyperparameters["batch_size"],
            verbose=0,
        )
        _, accuracy = self.model.evaluate(
            train_data.x_test, train_data.y_test, verbose=0
        )
        return accuracy


class EchoStateNetwork(Model):
    model: keras.Sequential
    hyperparameters: dict
    MODEL_HPARAMS = [
        "num_units",
        "leakage",
        "connectivity",
        "spectral_radius",
        "learning_rate",
    ]
    epochs: int

    def __init__(self, hyperparameters: dict, epochs: int):
        self.hyperparameters = hyperparameters
        self.epochs = epochs

        esn_layer = tfa.layers.ESN(
            units=self.hyperparameters["num_units"],
            connectivity=self.hyperparameters["connectivity"],
            leaky=self.hyperparameters["leakage"],
            spectral_radius=self.hyperparameters["spectral_radius"],
            use_norm2=False,
        )

        self.model = tf.keras.Sequential([esn_layer, keras.layers.Dense(units=1)])

        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.hyperparameters["learning_rate"]
            ),
            loss="mean_squared_error",
        )

    def load_data(test_size: float = 0.2):
        input_path = "data/temp_europa_2015-2019.nc"
        target_path = "data/targets.csv"

        x = xr.open_dataset(input_path)["temp"].to_numpy()
        y = genfromtxt(target_path, delimiter=",")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, shuffle=True
        )

        return TrainData(x_train, y_train, x_test, y_test)

    def evaluate(self, train_data: TrainData):
        self.model.fit(
            train_data.x_train, train_data.y_train, epochs=self.epochs, verbose=0
        )
        predictions = self.model(train_data.x_test)
        mse_loss = np.mean(tf.keras.losses.MSE(train_data.y_test, predictions))
        return mse_loss


class LeNet5(Model):
    model: keras.Sequential
    hyperparameters: dict
    MODEL_HPARAMS = [
        "learning_rate",
        "batch_size",
        "activation_function",
        "kernel_size",
        "number_of_filters",
    ]
    epochs: int

    def __init__(self, hyperparameters: dict, epochs: int):
        self.hyperparameters = hyperparameters
        self.epochs = epochs

        activation_function = self.hyperparameters.get("activation_function", "relu")
        kernel_size = self.hyperparameters.get("kernel_size", (5, 5))
        num_filters = self.hyperparameters.get("number_of_filters", [6, 16])

        self.model = keras.models.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28, 1)),
                keras.layers.Conv2D(
                    num_filters[0],
                    kernel_size=kernel_size,
                    activation=activation_function,
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(
                    num_filters[1],
                    kernel_size=kernel_size,
                    activation=activation_function,
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(120, activation=activation_function),
                keras.layers.Dense(84, activation=activation_function),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"]
        )
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def load_data(self, test_size: float = 0.2):
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
        x_test = x_test[..., tf.newaxis].astype("float32") / 255.0
        return TrainData(x_train, y_train, x_test, y_test)

    def evaluate(self, train_data: TrainData):
        self.model.fit(
            train_data.x_train,
            train_data.y_train,
            epochs=self.epochs,
            batch_size=self.hyperparameters["batch_size"],
            verbose=0,
        )
        _, accuracy = self.model.evaluate(
            train_data.x_test, train_data.y_test, verbose=0
        )
        return accuracy


class LucasModel(Model):
    hyperparameters: dict
    model: keras.Model
    MODEL_HPARAMS = ["Conv1D_filters", "kernel_size", "Dense1_units", "Dense2_units"]
    epochs: int

    def __init__(self, hyperparameters: dict, epochs: int):
        # Ensure all required hyperparameters are provided
        for hparam in LucasModel.MODEL_HPARAMS:
            if hparam not in hyperparameters:
                raise ValueError(f"Hyperparameter {hparam} not provided")
        self.hyperparameters = hyperparameters
        self.epochs = epochs

    def load_data(
        test_size: float = 0.2,
    ):
        input_path = "./data/lucas.csv"
        # Load data
        data = pd.read_csv(input_path)
        X = data.iloc[:, :-1].values  # All columns except the last one
        y = data.iloc[:, -1].values  # The last column

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Combine X and y for convenience
        train_combined = np.column_stack((X_train, y_train))
        test_combined = np.column_stack((X_test, y_test))

        # Remove rows with NaN values
        train_cleaned = train_combined[~np.isnan(train_combined).any(axis=1)]
        test_cleaned = test_combined[~np.isnan(test_combined).any(axis=1)]

        # Split them back into X and y
        X_train = train_cleaned[:, :-1]
        y_train = train_cleaned[:, -1]
        X_test = test_cleaned[:, :-1]
        y_test = test_cleaned[:, -1]

        return TrainData(X_train, y_train, X_test, y_test)

    def _build_model(self, inputs):
        hp = self.hyperparameters

        # prepare data for 1D CNN layer
        input_node = tf.nest.flatten(inputs)[0]
        input_node = tf.expand_dims(input_node, axis=2)

        # Ensure input shape's sequence length is valid
        if input_node.shape[1] <= hp["kernel_size"]:
            raise ValueError(
                f"Input shape's sequence length ({input_node.shape[1]}) is less than or equal to the kernel size ({hp['kernel_size']}). Consider using a smaller kernel size or ensuring your input data is correctly processed."
            )

        # model
        x = tf.keras.layers.Conv1D(
            hp["Conv1D_filters"],
            hp["kernel_size"],
            activation="LeakyReLU",
            input_shape=input_node.shape[1:],
        )(input_node)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(hp["Dense1_units"], activation="LeakyReLU")(x)
        output_node = tf.keras.layers.Dense(hp["Dense2_units"], activation="LeakyReLU")(
            x
        )

        return output_node

    def evaluate(self, train_data: TrainData):
        # Construct the model
        inputs = tf.keras.Input(shape=(train_data.x_train.shape[1],))
        outputs = self._build_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        model.fit(
            train_data.x_train,
            train_data.y_train,
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
        )

        # Evaluation
        _, mae = model.evaluate(train_data.x_test, train_data.y_test, verbose=0)

        return mae
