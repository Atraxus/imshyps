import os
from abc import ABC, abstractmethod
from enum import Enum

import xarray as xr
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}


# This needs to be here to avoid stdout spam from tensorflow
import tensorflow as tf  # trunk-ignore(ruff/E402)
import tensorflow_addons as tfa  # trunk-ignore(ruff/E402)
from tensorflow import keras  # trunk-ignore(ruff/E402)


class TrainData:  # TODO: Remove validation set
    x_train: tf.Tensor
    y_train: tf.Tensor
    x_val: tf.Tensor
    y_val: tf.Tensor
    x_test: tf.Tensor
    y_test: tf.Tensor

    def __init__(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_val: tf.Tensor,
        y_val: tf.Tensor,
        x_test: tf.Tensor,
        y_test: tf.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


class EvalMetrics(Enum):
    ACCURACY = "accuracy"
    BINARY_ACCURACY = "binary_accuracy"
    CATEGORICAL_ACCURACY = "categorical_accuracy"
    BINARY_CROSSENTROPY = "binary_crossentropy"
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
    AUC = "AUC"


class Model(ABC):
    # List of hyperparameters that the model needs. These will be read from
    # the config file. If a config file does not contain all of these, it is
    # invalid. Parameters that are not in this list will be ignored.
    MODEL_HPARAMS = []

    @abstractmethod
    # Initialize the model with the given hyperparameters. It is a dict with
    # key-value pairs. The default metric used for evaluation is accuracy.
    def __init__(self, hyperparameters: dict, metrics: list = None):
        if metrics is None:
            metrics = [EvalMetrics.ACCURACY]
        pass

    @abstractmethod
    # Train the model and return the accuracy
    def load_data(
        input_path: str = None, target_path: str = None, test_size: float = 0.2
    ):
        pass

    @abstractmethod
    # Train the model and return the accuracy
    def evaluate(self, train_data: TrainData):
        pass


class MLP(Model):
    model: keras.Sequential
    metrics: list
    hyperparameters: dict
    MODEL_HPARAMS = [
        "learning_rate",
        "batch_size",
        "activation_function",
        "num_layers",
        "num_neurons",
    ]

    def __init__(self, hyperparameters: dict, metrics: list = None):
        if metrics is None:
            metrics = [EvalMetrics.ACCURACY]
        self.hyperparameters = hyperparameters
        self.metrics = metrics

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
            metrics=metrics,
            learning_rate=self.hyperparameters["learning_rate"],
        )

    # Get the TrainData object from the MNIST dataset
    def load_data(
        input_path: str = None, target_path: str = None, test_size: float = 0.2
    ):
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize the data
        x_train = keras.utils.normalize(x_train, axis=1)
        x_test = keras.utils.normalize(x_test, axis=1)

        # split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=test_size  # TODO: This does not make sense
        )

        return TrainData(x_train, y_train, x_val, y_val, x_test, y_test)

    # Train the model and return the accuracy
    def evaluate(self, train_data: TrainData):
        self.model.fit(
            train_data.x_train,
            train_data.y_train,
            epochs=10,
            batch_size=self.hyperparameters["batch_size"],
            validation_data=(train_data.x_val, train_data.y_val),
        )
        _, accuracy = self.model.evaluate(train_data.x_test, train_data.y_test)
        return accuracy


class EchoStateNetwork(Model):
    model: keras.Sequential
    metrics: list
    hyperparameters: dict
    MODEL_HPARAMS = [
        "num_units",
        "leakage",
        "connectivity",
        "spectral_radius",
        "learning_rate",
    ]

    def __init__(self, hyperparameters: dict, metrics: list = None):
        if metrics is None:
            metrics = [EvalMetrics.ACCURACY]
        self.hyperparameters = hyperparameters
        self.metrics = metrics

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
            # metrics=[tf.keras.metrics.Accuracy()], # TODO
        )

    def load_data(input_path: str, target_path: str, test_size: float = 0.2):
        x = xr.open_dataset(input_path)["temp"].to_numpy()
        y = genfromtxt(target_path, delimiter=",")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, shuffle=True
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=test_size, shuffle=True
        )
        return TrainData(x_train, y_train, x_val, y_val, x_test, y_test)

    def evaluate(self, train_data: TrainData):
        self.model.fit(train_data.x_train, train_data.y_train, epochs=1000)
        mse_loss = self.model.evaluate(train_data.x_test, train_data.y_test)
        return mse_loss


class CNN(Model):
    ...


class GFZ_CNN(Model):
    # input_1 (InputLayer)
    # cast_to_float32 (CastToFloat32)
    # expand_dims (TFOpLambda)
    # conv1d (Conv1D)
    # flatten (Flatten)
    # dense (Dense)
    # dense_1 (Dense)
    # dropout (Dropout)
    # regression_head_1 (Dense)
    model: keras.Sequential
    metrics: list
    hyperparameters: dict
    MODEL_HPARAMS = [
        "learning_rate",
        "optimizer",
        "classification_head_1/dropout",
        "structured_data_block_1/dense_block_1/units_1",
        "structured_data_block_1/dense_block_1/dropout",
        "structured_data_block_1/dense_block_1/units_0",
        "structured_data_block_1/dense_block_1/num_layers",
        "structured_data_block_1/dense_block_1/use_batchnorm",
        "structured_data_block_1/normalize",
    ]

    def __init__(self, hyperparameters: dict, metrics: list = None):
        if metrics is None:
            metrics = [EvalMetrics.ACCURACY]
        self.hyperparameters = hyperparameters
        self.metrics = metrics

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.InputLayer(input_shape=(28, 28)))
        self.model.add(keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)))
        self.model.add(keras.layers.Conv1D(32, 3, activation="relu"))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation="relu"))
        self.model.add(keras.layers.Dense(10, activation="softmax"))

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"]
        )
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics,
        )

    # Train the model and return the accuracy
    def evaluate(self, train_data: TrainData):
        self.model.fit(
            train_data.x_train,
            train_data.y_train,
            epochs=1,
            batch_size=self.hyperparameters["batch_size"],
            validation_data=(train_data.x_val, train_data.y_val),
        )
        _, accuracy = self.model.evaluate(train_data.x_test, train_data.y_test)
        return accuracy
