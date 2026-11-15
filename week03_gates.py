import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)

for name, y in {
    "AND": [0,0,0,1],
    "OR":  [0,1,1,1],
    "XOR": [0,1,1,0]
}.items():
    model = Sequential([Dense(4, activation="relu", input_shape=(2,)),
                        Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(X, np.array(y), epochs=500, verbose=0)
    print(name, np.round(model.predict(X, verbose=0).ravel()).astype(int))
