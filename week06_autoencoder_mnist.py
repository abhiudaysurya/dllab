import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

inp = layers.Input((784,))
encoded = layers.Dense(64, activation="relu")(inp)
decoded = layers.Dense(784, activation="sigmoid")(encoded)
autoencoder = Model(inp, decoded)
autoencoder.compile("adam", "binary_crossentropy")
autoencoder.fit(x_train.reshape(-1, 784), x_train.reshape(-1, 784),
                epochs=5, batch_size=256, validation_split=0.1)

recon = autoencoder.predict(x_test[:5].reshape(-1, 784))
for i in range(5):
    plt.subplot(2, 5, i+1); plt.imshow(x_test[i], cmap="gray"); plt.axis("off")
    plt.subplot(2, 5, i+6); plt.imshow(recon[i].reshape(28, 28), cmap="gray"); plt.axis("off")
plt.show()
