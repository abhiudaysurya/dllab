import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = x[:1000].astype("float32") / 255.
# Fixed Sobel-like edge kernel implemented as a convolution filter.
edge = layers.Conv2D(1, 3, padding="same", use_bias=False, trainable=False)
edge.build((None, 28, 28, 1))
edge.set_weights([tf.constant([[[[-1.]], [[0.]], [[1.]]],
                                [[[-2.]], [[0.]], [[2.]]],
                                [[[-1.]], [[0.]], [[1.]]]]) .numpy()])
filtered = edge(x[..., None]).numpy()
for i in range(5):
    plt.subplot(2, 5, i+1); plt.imshow(x[i], cmap="gray"); plt.axis("off")
    plt.subplot(2, 5, i+6); plt.imshow(abs(filtered[i, :, :, 0]), cmap="gray"); plt.axis("off")
plt.show()

# CNN can learn image features from original and filtered images.
model = Sequential([layers.Input((28,28,1)), layers.Conv2D(16,3,activation="relu"),
                     layers.MaxPooling2D(), layers.Flatten(), layers.Dense(10,activation="softmax")])
(_, y_train), _ = tf.keras.datasets.mnist.load_data()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x[...,None], y_train[:1000], epochs=2)
