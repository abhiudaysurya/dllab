import tensorflow as tf
from tensorflow.keras import layers, Model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.
base = tf.keras.applications.MobileNetV2(
    input_shape=(96,96,3), include_top=False, weights="imagenet")
base.trainable = False

inp = layers.Input((32,32,3))
x = layers.Resizing(96,96)(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(10, activation="softmax")(x)
model = Model(inp, out)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, validation_split=0.1, batch_size=64)
print("Test accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])
