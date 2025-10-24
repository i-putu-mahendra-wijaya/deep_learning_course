"""
Multi-Label Image Classification using Convolutional Neural Networks (CNN)

This script demonstrates the use of deep learning for a multi-label image
classification task, where each image is associated with multiple labels.
Specifically, we train a CNN model to classify both the gender and the
style or usage category of watch images.

The dataset used in this example is the **Fashion Product Images (Small)**
dataset, available from Kaggle:
    https://www.kaggle.com/paramaggarwal/fashion-product-images-small

It is assumed that the dataset has been downloaded and extracted to:
    ~/.keras/datasets/fashion-product-images-small/

History:
    - 2025-10-24 | I Putu Mahendra Wijaya | Initial creation
"""


from typing import Tuple, List, Dict, Set
from pprint import pprint
import os
import pathlib
from PIL import ImageFile
from pathlib import Path
from csv import DictReader
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import *

def load_images_and_labels(
        image_paths: List[Path],
        styles: Dict,
        target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset of images and their corresponding labels.

    This function loads images from a list of file paths, converts each image
    into a NumPy array, and extracts labels based on the provided `styles`
    metadata. The output is a tuple of NumPy arrays containing the image data
    and their associated labels.

    :param image_paths: List of file paths to the image files.
    :type image_paths: List[str]

    :param styles: Dictionaries, each describing metadata such as
        gender and style for the corresponding image.
    :type styles: Dict[str, Dict]

    :param target_size: Desired image dimensions (width, height) for resizing.
    :type target_size: Tuple[int, int]

    :return: A tuple ``(images, labels)`` where:
        - ``images`` is a NumPy array of shape ``(n_samples, height, width, channels)``
          containing the image data.
        - ``labels`` is a NumPy array containing the corresponding label values.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    images: List[np.ndarray] = []
    labels: List[Tuple] = []

    for each_image_path in image_paths:

        img_image: ImageFile = load_img(
            path=each_image_path,
            target_size=target_size,
        )

        img_arr: np.ndarray = img_to_array(img_image)
        image_id = str(each_image_path).split(os.path.sep)[-1][:-4]

        img_style: Dict[str, Dict] = styles[image_id]

        labels_tpl: Tuple = (
            img_style["gender"],
            img_style["usage"],
        )

        images.append(img_arr)
        labels.append(labels_tpl)

    return np.array(images), np.array(labels)


def build_network(
        width: int,
        height: int,
        depth: int,
        classes: int
) -> Model:
    """
    Build a Convolutional Neural Network (CNN) model for multi-label classification.

    This function constructs a CNN architecture tailored for multi-label classification tasks.
    The model expects input images with specified width, height, and depth (number of channels),
    and produces output predictions across a defined number of label classes.

    :param width: Width of the input image in pixels.
    :type width: int

    :param height: Height of the input image in pixels.
    :type height: int

    :param depth: Number of color channels in the input image (e.g., 1 for grayscale, 3 for RGB).
    :type depth: int

    :param classes: Number of label classes to predict.
    :type classes: int

    :return: A compiled Keras ``Model`` instance representing the CNN architecture.
    :rtype: keras.Model
    """

    input_layer: Input = Input(shape=(height, width, depth))
    conv_1: Conv2D = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
    )(input_layer)
    activation_1: ReLU = ReLU()(conv_1)
    batch_norm_1: BatchNormalization = BatchNormalization(axis=-1)(activation_1)

    conv_2: Conv2D = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
    )(batch_norm_1)
    activation_2: ReLU = ReLU()(conv_2)
    batch_norm_2: BatchNormalization = BatchNormalization(axis=-1)(activation_2)

    max_pool_1: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2)
    )(batch_norm_2)
    dropout_1: Dropout = Dropout(rate=0.25)(max_pool_1)

    conv_3: Conv2D = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
    )(dropout_1)
    activation_3: ReLU = ReLU()(conv_3)
    batch_norm_3: BatchNormalization = BatchNormalization(axis=-1)(activation_3)

    conv_4: Conv2D = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
    )(batch_norm_3)
    activation_4: ReLU = ReLU()(conv_4)
    batch_norm_4: BatchNormalization = BatchNormalization(axis=-1)(activation_4)

    max_pool_2: MaxPooling2D = MaxPooling2D(
        pool_size=(2, 2)
    )(batch_norm_4)
    dropout_2: Dropout = Dropout(rate=0.25)(max_pool_2)

    flatten_1: Flatten = Flatten()(dropout_2)
    dense_1: Dense = Dense(
        units=512
    )(flatten_1)
    activation_5: ReLU = ReLU()(dense_1)
    batch_norm_5: BatchNormalization = BatchNormalization(axis=-1)(activation_5)
    dropout_3: Dropout = Dropout(rate=0.25)(batch_norm_5)

    dense_2: Dense = Dense(
        units=classes
    )(dropout_3)

    output: Activation = Activation("sigmoid")(dense_2)

    return Model(
        inputs=input_layer
        , outputs=output
    )


if __name__ == "__main__":

    SEED: int = 42 # set the random to guarantee reproducibility
    np.random.seed(SEED)

    """
    Folder structure of fashion-product-images-small dataset
    
    fashion-product-images-small
    ├── images
    └── myntradataset
        └── ...
    
    """

    base_path: Path = (
        pathlib.Path.home() / ".keras" / "datasets" / "fashion-product-images-small"
    )

    style_path: Path = (
        base_path / "styles.csv"
    )

    images_path_pattern: Path = (
        base_path / "images" / "*.jpg"
    )

    image_paths: List[Path] = [
        Path(each_str) for each_str
        in [*glob.glob(str(images_path_pattern))]
    ]

    # We focus only `Watches` images for `Causal` , `Smart Causal`, and `Formal` usages
    # suited fo `Men` and `Women`
    with open(style_path, mode="r") as style_file_handle:
        dict_reader: DictReader = DictReader(style_file_handle)

        styles_ : List[Dict] = [*dict_reader]

        article_type: str = "Watches"
        genders: Set = {"Men", "Women"}
        usages: Set = {"Causal", "Smart Causal", "Formal"}

        style_dict: Dict = {
            each_style["id"]: each_style
            for each_style in styles_
            if (
                each_style["articleType"] == article_type
                and
                each_style["gender"] in genders
                and
                each_style["usage"] in usages
            )
        }

        watch_image_paths: List[Path] = [
            *filter(
                lambda pth: str(pth).split(os.path.sep)[-1][:-4] in style_dict.keys(), image_paths)
        ]

        pprint(style_dict, indent=4)
        pprint(watch_image_paths, indent=4)