"""
This script is to show students how to visualize
a model architecture using Keras

There are two ways to visualize a model architecture:
1. Using the model.summary() method
2. Using the model.plot() method

History:
- 2025 October 20 | I Putu Mahendra Wijaya | Initial creation
"""

from PIL import Image
from pprint import pprint
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.utils import plot_model



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
    input_layer: Input = Input(shape=(64, 64, 1), name="input_layer")

    # Defining the CNN layers
    conv_1: Conv2D = Conv2D(
        kernel_size=(2, 2),
        padding="same",
        strides=(2, 2),
        filters=32,
        name="conv_1"
    )(input_layer)
    activation_1: LeakyReLU = LeakyReLU(name="activation_1")(conv_1)
    batch_norm_1: BatchNormalization = BatchNormalization(name="batch_norm_1")(activation_1)
    pooling_1: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        name="pooling_1"
    )(batch_norm_1)
    conv_2: Conv2D = Conv2D(
        kernel_size=(2, 2),
        padding="same",
        strides=(2, 2),
        filters=64,
        name="conv_2"
    )(pooling_1)
    activation_2: LeakyReLU = LeakyReLU(name="activation_2")(conv_2)
    batch_norm_2: BatchNormalization = BatchNormalization(name="batch_norm_2")(activation_2)
    pooling_2: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding="same",
        name="pooling_2"
    )(batch_norm_2)
    dropout = Dropout(rate=0.5, name="dropout")(pooling_2)

    # Defining the Dense layers
    flatten: Flatten = Flatten(name="flatten")(dropout)
    dense_1: Dense = Dense(
        units=256,
        name="dense_1"
    )(flatten)
    activation_3: LeakyReLU = LeakyReLU(name="activation_3")(dense_1)
    dense_2: Dense = Dense(
        units=128,
        name="dense_2"
    )(activation_3)
    activation_4: LeakyReLU = LeakyReLU(name="activation_4")(dense_2)
    dense_3: Dense = Dense(
        units=3,
        name="dense_3"
    )(activation_4)
    output: Softmax = Softmax(name="output")(dense_3)

    model: Model = Model(
        inputs=input_layer,
        outputs=output,
        name="my_model"
    )

    return model


if __name__ == "__main__":

    # Build the model
    model: Model = build_model()

    # Show model architecture using text summary
    pprint(model.summary(), indent=4)

    # plot a diagram of the model architecture
    plot_model(
        model,
        to_file="model_architecture.png",
        show_shapes=True
    )
    model_diag = Image.open("model_architecture.png")
    model_diag.show()