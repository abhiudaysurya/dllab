import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# CNN character/digit recognition
cnn = models.Sequential([
    layers.Input((28, 28, 1)), layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(), layers.Flatten(),
    layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax")
])
cnn.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x_train[..., None], y_train, epochs=2, validation_split=0.1)
print("CNN test accuracy:", cnn.evaluate(x_test[..., None], y_test, verbose=0)[1])

# RNN: treat each image row as one time step
rnn = models.Sequential([
    layers.Input((28, 28)), layers.SimpleRNN(64),
    layers.Dense(10, activation="softmax")
])
rnn.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
rnn.fit(x_train, y_train, epochs=2, validation_split=0.1)
print("RNN test accuracy:", rnn.evaluate(x_test, y_test, verbose=0)[1])
