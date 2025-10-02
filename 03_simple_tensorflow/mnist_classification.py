"""
Date: 2025 October 02
Author: I Putu Mahendra Wijaya

This script is a simple example of building a neural network using TensorFlow / Keras

The task at hand can be categorized as a classification task.

We will use the MNIST dataset to train a simple neural network.

This script demonstrates the following:
1. Loading the MNIST dataset
2. Building a simple neural network using Keras
3. Training the neural network
4. Evaluating the neural network
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pprint import pprint

#  Ref: https://www.tensorflow.org/datasets/catalog/mnist
#  MNIST is a standard dataset for computer vision.
#  The dataset consists of 60,000 training images and 10,000 test images.
#  Each image is 28x28 pixels, and each pixel is represented by a grayscale value (0-255).

# 1. Load MNIST dataset (MMIST digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# -- Normalize value to 0-1 range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# -- Flatten from 28x28 to 784
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


# 2. Build a simple neural network
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),                                      # input layer
        layers.Dense(512, activation="relu", input_shape=(784,)), # hidden layer
        layers.Dense(10, activation="softmax"),                   # output layer
    ]
)


# 3. Compile model (optimizer, loss, metrics)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# 4. Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)


# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
pprint(f"Test accuracy: {test_acc:.2f}")