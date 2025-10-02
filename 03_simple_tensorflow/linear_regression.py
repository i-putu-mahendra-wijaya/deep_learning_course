"""
Date: 2025 October 02
Author: I Putu Mahendra Wijaya

This script is a simple example of building a neural network using TensorFlow / Keras

The task at hand can be categorized as a regression task.

We will use a toy `y=2x + 1` linear equation to train a simple neural network.

This script demonstrates the following:
1. Create a synthetic dataset using NumPy with random noise
2. Building a simple neural network using Keras
3. Training the neural network
4. Evaluating the neural network
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pprint import pprint


# 1. Create toy data
# y = 2x + 1 with some noise
x = np.linspace(-1, 1, 100)
y = 2 * x + 1 + np.random.normal(0, 0.1, size=x.shape)


# -- Reshape for Keras (expecting 2D inpt: (samples, features))
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# 2. Build a simple neural network
model = keras.Sequential(
    [
        keras.Input(shape=(1,)),                                            # input layer
        layers.Dense(10, activation="relu"),                          # hidden layer
        layers.Dense(1, activation="linear"),                         # output layer
    ]
)

# 3. Compile the model
model.compile(
    optimizer="sgd",
    loss="mse",
    metrics=["mse"],
)

# 4. Train the model
model.fit(x, y, epochs=100, verbose=1)


# 5. Check the learned weights
weights, bias = model.layers[0].get_weights()
pprint(f"Weights: {weights[0][0]:.2f}")
pprint(f"Bias: {bias[0]:.2f}")


# 6. Predict
x_test = np.array([[0.5]])
y_pred = model.predict(x_test)
pprint(f"Predicted for x:0.5; y: {y_pred[0][0]:.2f}")