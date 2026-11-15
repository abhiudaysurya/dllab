import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

(x, y), _ = tf.keras.datasets.mnist.load_data()
x = x[:5000].astype("float32") / 255.
inp = layers.Input((784,))
z = layers.Dense(32, activation="relu", name="features")(inp)
out = layers.Dense(784, activation="sigmoid")(z)
ae = Model(inp, out)
ae.compile("adam", "binary_crossentropy")
ae.fit(x.reshape(-1,784), x.reshape(-1,784), epochs=5, batch_size=256, verbose=1)

encoder = Model(inp, ae.get_layer("features").output)
features = encoder.predict(x.reshape(-1,784), verbose=0)
labels = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(features)
plt.scatter(features[:,0], features[:,1], c=labels, s=5)
plt.title("K-Means clusters of compressed MNIST images")
plt.show()
