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

from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

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


# 6. Show grid of MNIST pictures
def show_grid_mnist(
        model: keras.Sequential,
        x_test: np.ndarray,
        y_test: np.ndarray,
        grid_size: Tuple[int, int] = (3, 3)
) -> None:

    """
    Show grid of MNIST picture to allow user see how the model
    is performing visually

    :param model: a keras model
    :param x_test: mnist test images
    :param y_test: mnist test labels
    :param grid_size: size of grid. defaults to (3, 3)
    :return: None
    """

    num_images: int = grid_size[0] * grid_size[1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8))
    fig.suptitle("MNIST Classification")

    axes = axs.flatten()

    for loop_idx in range(num_images):
        # select a random index
        rnd_idx = np.random.randint(0, x_test.shape[0])

        # Get the image and its true label
        img = x_test[rnd_idx]
        true_label = y_test[rnd_idx]

        # Reshape the image back 28x28 for showing
        img_reshaped = img.reshape(28, 28)

        # Predict the label using the trained model
        pred = model.predict(img[np.newaxis, ...], verbose=0)
        pred_label = np.argmax(pred)

        # Display the image in the current subplot
        axes[loop_idx].imshow(img_reshaped, cmap="gray")
        axes[loop_idx].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[loop_idx].axis("off")

    plt.tight_layout()
    plt.show()

show_grid_mnist(model, x_test, y_test)