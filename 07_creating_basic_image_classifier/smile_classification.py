"""
Smile Detection using TensorFlow Keras

This script demonstrates how to build and train a simple binary classification
model to detect whether a person is smiling in a given photo. The model is
implemented using TensorFlow Keras and trained on the **SMILEsmileD** dataset.

Dataset:
    The SMILEsmileD dataset can be obtained from:
    https://github.com/hromi/SMILEsmileD.git

    It is assumed that the dataset has been downloaded and extracted into:
    `./keras/dataset/SMILEsmileD-master`

References:
    - Hromi, M. (2014). *SMILEsmileD Dataset*. GitHub repository.

History:
    - 2025-10-23 | I Putu Mahendra Wijaya | Initial creation
"""


from typing import Tuple, List
from pprint import pprint
from PIL import ImageFile
from pathlib import Path
import pathlib
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *


def load_images_and_labels(
    img_paths: List[Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and extract corresponding numeric labels.

    This function takes a list of image paths, loads each image in grayscale
    and resizes it to 32×32 pixels. The label is derived from the parent
    directory name, with `"positive"` mapped to 1.0 and everything else to 0.0.

    :param img_paths: List of image file paths to load.
    :type img_paths: List[Path]
    :returns: A tuple containing two numpy arrays:
              - images: Array of shape (N, 32, 32, 1)
              - labels: Array of numeric labels (float)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    images: List[np.ndarray] = []
    labels: List[float] = []

    for each_image_path in img_paths:
        img_file: ImageFile = load_img(
            each_image_path,
            target_size=(32,32),
            color_mode="grayscale"
        )
        img_arr: np.ndarray = img_to_array(img_file)

        label: str = str(each_image_path).split(os.path.sep)[-2]
        is_positive: bool = "positive" in label
        numeric_label: float = float(is_positive)

        images.append(img_arr)
        labels.append(numeric_label)

    return np.array(images), np.array(labels)


def build_network(

) -> Model:
    """
    Build Deep Learning Network

    This function define the deep learning network to classify the images.
    The network is consisted of:
    - two convolutional layer with ELU activation function
    - batch normalization layer to stabilize the model
    - dropout layer to reduce over-fitting
    - one dense layer

    The output layer uses `sigmoid` activation function, because the network
    would perform binary classification: is_smile, or not_smile

    :return: A Keras Model
    :rtype: Model
    """

    input_layer: Input = Input(shape=(32, 32, 1))
    conv_1: Conv2D = Conv2D(
        filters=20,
        kernel_size=(5, 5),
        padding="same",
        strides=(1, 1),
    )(input_layer)
    activation_1: ELU = ELU()(conv_1)
    batch_norm_1: BatchNormalization = BatchNormalization()(activation_1)
    max_pool_1: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(batch_norm_1)

    drop_out_1: Dropout = Dropout(0.5)(max_pool_1)

    conv_2: Conv2D = Conv2D(
        filters=40,
        kernel_size=(5, 5),
        padding="same",
        strides=(1, 1),
    )(drop_out_1)
    activation_2: ELU = ELU()(conv_2)
    batch_norm_2: BatchNormalization = BatchNormalization()(activation_2)
    max_pool_2: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(batch_norm_2)
    drop_out_2: Dropout = Dropout(0.5)(max_pool_2)
    flat_1: Flatten = Flatten()(drop_out_2)
    dense_1: Dense = Dense(units=500)(flat_1)
    activation_3: ELU = ELU()(dense_1)
    drop_out_3: Dropout = Dropout(0.5)(activation_3)

    output_layer: Dense = Dense(units=1, activation="sigmoid")(drop_out_3)

    return Model(
        inputs=input_layer,
        outputs=output_layer,
    )


if __name__ == "__main__":

    """
    Folder Structure of Smile Dataset
    
    SMILEsmileD-master
    ├── SMILEs
    │     ├── negatives
    │     │     └── negatives7
    │     └── positives
    │           └── positives7
    ├── ...
        
    """
    img_file_path: Path = (
        pathlib.Path.home() / ".keras" / "datasets" / "SMILEsmileD-master" / "SMILEs" / "**" / "*.jpg"
    )
    dataset_paths: List[Path] = [
        Path(each_str) for each_str in
        [*glob.glob(str(img_file_path), recursive=True)]
    ]

    img_arr, label_arr = load_images_and_labels(img_paths=dataset_paths)

    # Normalize the images and compute the number of positive, negative, and count_dataset
    img_arr = img_arr / 255.0
    count_dataset: int = len(label_arr)
    count_positive: int = np.sum(label_arr == 1)
    count_negative: int = np.sum(label_arr == 0)

    pprint(
        f"""
        count_dataset: {count_dataset}
        count_positive: {count_positive}
        count_negative: {count_negative}
        """
        , indent=4
    )

    # Create train, test, and validation subsets
    (
        img_train,
        img_test,
        label_train,
        label_test
    ) = train_test_split(
        img_arr,
        label_arr,
        test_size=0.2,
        stratify=label_arr,
        random_state=42,
    )

    (
        img_train,
        img_val,
        label_train,
        label_val
    ) = train_test_split(
        img_train,
        label_train,
        test_size=0.2,
        stratify=label_train,
        random_state=42,
    )


    # Instantiate and compile the model
    cnn_model: Model = build_network()
    cnn_model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )


    # Train the model
    # Because the dataset is unbalanced, we are
    # assigning each class proportional weights according to the count_positive / count_negative
    # in the dataset
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    cnn_model.fit(
        img_train,
        label_train,
        validation_data=(img_val, label_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight={
            1.0 : count_dataset / count_positive,
            0.0 : count_dataset / count_negative,
        },
        verbose=2,
    )


    # Evaluate the model on the test dataset
    test_loss, test_acc = cnn_model.evaluate(img_test, label_test)

    pprint(
    f"""
    test_loss: {test_loss}
    test_acc: {test_acc}
    """
    )