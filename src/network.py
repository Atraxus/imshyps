import tensorflow.keras as keras
import tensorflow as tf

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



class TestNet:
    model: keras.Sequential

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
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Train function uses given learning_rate and batch_size
    def train(self, train_data: TrainData):
        history = self.model.fit(
            train_data.x_train, train_data.y_train, epochs=3, batch_size=self.batch_size, validation_data=(train_data.x_val, train_data.y_val)
        )
        test_loss, test_acc = self.model.evaluate(train_data.x_test, train_data.y_test)
        
        # Get validation loss from history object
        validation_loss = history.history['val_loss']
        
        return test_loss, test_acc, validation_loss
