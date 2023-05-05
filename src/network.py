import tensorflow.keras as keras
import tensorflow as tf


class TestNetwork():
    model: keras.Sequential
    x_train: tf.Tensor
    y_train: tf.Tensor
    x_test: tf.Tensor
    y_test: tf.Tensor

    learning_rate: float
    batch_size: int
    activation_function: str


    def __init__(self, learning_rate: float, batch_size: int, activation_function: str):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation_function = activation_function

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation=self.activation_function))
        self.model.add(keras.layers.Dense(64, activation=self.activation_function))
        self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


    def load_data(self):
        mnist = keras.datasets.mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.x_train = keras.utils.normalize(self.x_train, axis=1)
        self.x_test = keras.utils.normalize(self.x_test, axis=1)
        

    # Train function uses given learning_rate and batch_size
    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=self.batch_size)
        val_loss, val_acc = self.model.evaluate(self.x_test, self.y_test)
        return val_loss, val_acc