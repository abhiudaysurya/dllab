# Mini-project example: image classification pipeline using CIFAR-10.
import tensorflow as tf
from tensorflow.keras import layers, Sequential

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.
model = Sequential([
    layers.Input((32,32,3)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)
print("Test results:", model.evaluate(x_test, y_test, verbose=0))
model.save("mini_project_model.keras")

# Predict one test image
prediction = model.predict(x_test[:1], verbose=0).argmax()
print("Predicted class:", prediction, "| Actual class:", int(y_test[0]))
