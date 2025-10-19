"""
This script is an example to show students that
there are more than one ways to solve a task

In this case, we will show how to classify MNIST dataset using CNN

The script demonstrates the following:
1. Loading the MNIST dataset
2. Building a CNN using Keras
3. Training the CNN
4. Evaluating the CNN
5. Saving the model
6. Loading the model

History:
- 2025 October 19 | I Putu Mahendra Wijaya | Initial creation
"""

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ReLU, Softmax
from tensorflow.keras.models import load_model


def load_data(

) -> Tuple:
    """
    Load MNIST dataset and then prepare the data by:
    * normalizing the data
    * one-hot encoding the labels

    :param:
    - None

    :return:
    A tuple containing:
        - X_train: training data
        - X_test: testing data
        - y_train: training labels
        - y_test: testing labels
    """

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the data
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape greyscale to include channel dimension
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Process labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)

    return X_train, X_test, y_train, y_test


def build_model(

) -> Model:
    """
    Function to build the CNN model.
    The architecture is comporised of:
        * a single convolutional layer
        * two fully connected (dense) layers

    :param:
    - None

    :return:
    A Keras Model instance
    """
    input_layer: Input = Input(shape=(28, 28, 1))
    conv_1: Conv2D = Conv2D(
        kernel_size=(2, 2),
        padding="same",
        strides=(2, 2),
        filters=32
    )(input_layer)
    activation_1: ReLU = ReLU()(conv_1)
    batch_norm_1: BatchNormalization = BatchNormalization()(activation_1)
    pooling_1: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1)
    )(batch_norm_1)
    dropout = Dropout(0.5)(pooling_1)

    flatten: Flatten = Flatten()(dropout)
    dense_1: Dense = Dense(units=128)(flatten)
    activation_2: ReLU = ReLU()(dense_1)
    dense_2: Dense = Dense(units=10)(activation_2)
    output: Softmax = Softmax()(dense_2)

    model: Model = Model(
        inputs=input_layer,
        outputs=output
    )

    return model


def evaluate_model(
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> None:
    """

    Function to evaluate the model.

    In this context, the evaluation is done using the accuracy metric.

    :param model:
    :param X_test:
    :param y_test:

    :return: None
    """

    _, accur = model.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy: {accur: .2f}")


if __name__ == "__main__":

    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Build the model
    model: Model = build_model()

    # Compile the model, and then train the model
    # Feel free to tune the epochs and the batch_size to suit your machine capacity
    count_epochs: int = 50
    batch_size: int = 1024

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=count_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # Saving the model and weights into native keras file
    model.save("mnist_model_and_weights.keras")

    # Load the model and weights from keras file
    model: Model = load_model("mnist_model_and_weights.keras")

    # Evaluate the model loaded from HDF5 file
    evaluate_model(model, X_test, y_test)