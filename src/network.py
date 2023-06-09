import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import tensorflow.keras as keras
import tensorflow as tf

from enum import Enum
from sklearn.model_selection import train_test_split

class TrainData:
    x_train: tf.Tensor
    y_train: tf.Tensor
    x_val: tf.Tensor
    y_val: tf.Tensor
    x_test: tf.Tensor
    y_test: tf.Tensor

    def __init__(self, val_size=0.2):
        mnist = keras.datasets.mnist

        (x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # normalize the data
        x_train = keras.utils.normalize(x_train, axis=1)
        self.x_test = keras.utils.normalize(self.x_test, axis=1)

        # split training data into training and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, self.y_train, test_size=val_size
        )


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


class MLP:
    model: keras.Sequential
    metrics: list
    hyperparameters: dict
    MODEL_HPARAMS = ["learning_rate", "batch_size",
                     "activation_function", "num_layers", "num_neurons"]

    def __init__(self, hyperparameters: dict, metrics: list = [EvalMetrics.ACCURACY]):
        self.hyperparameters = hyperparameters
        self.metrics = metrics

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Flatten())
        for _ in range(self.hyperparameters["num_layers"]):
            self.model.add(keras.layers.Dense(
                self.hyperparameters["num_neurons"], activation=self.hyperparameters["activation_function"]))
        self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

        optimizer = keras.optimizers.Adam(
            learning_rate=self.hyperparameters["learning_rate"])
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
            epochs=1, # TODO: make this a hyperparameter
            batch_size=self.hyperparameters["batch_size"],
            validation_data=(train_data.x_val, train_data.y_val),
        )
        _, accuracy = self.model.evaluate(
            train_data.x_test, train_data.y_test)
        return accuracy
