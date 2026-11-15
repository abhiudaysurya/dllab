import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.astype("float32") - 127.5) / 127.5
x = x.reshape(-1, 784)
G = tf.keras.Sequential([layers.Input((100,)), layers.Dense(256, activation="relu"),
                         layers.Dense(512, activation="relu"),
                         layers.Dense(784, activation="tanh")])
D = tf.keras.Sequential([layers.Input((784,)), layers.Dense(256, activation="relu"),
                         layers.Dense(1, activation="sigmoid")])
opt_g, opt_d = tf.keras.optimizers.Adam(0.0002), tf.keras.optimizers.Adam(0.0002)
bce = tf.keras.losses.BinaryCrossentropy()

for epoch in range(5):
    for real in tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(128):
        noise = tf.random.normal((tf.shape(real)[0], 100))
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(noise, training=True)
            loss_d = bce(tf.ones((len(real), 1)), D(real, training=True)) + \
                     bce(tf.zeros((len(real), 1)), D(fake, training=True))
            loss_g = bce(tf.ones((len(real), 1)), D(fake, training=True))
        opt_d.apply_gradients(zip(dt.gradient(loss_d, D.trainable_variables), D.trainable_variables))
        opt_g.apply_gradients(zip(gt.gradient(loss_g, G.trainable_variables), G.trainable_variables))
    print("Epoch", epoch + 1, "Generator loss:", float(loss_g))

imgs = G(tf.random.normal((10, 100))).numpy().reshape(-1, 28, 28)
for i, img in enumerate(imgs):
    plt.subplot(2, 5, i+1); plt.imshow(img, cmap="gray"); plt.axis("off")
plt.show()
