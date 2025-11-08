"""
Data Augmentation for Enhancing Model Generalization.

In many practical scenarios, machine learning practitioners must train models on
limited datasets. Such constraints often lead to overfitting, where the model
memorizes patterns from the small training set rather than learning to generalize
to unseen data.

Data augmentation provides an effective strategy to mitigate overfitting by
artificially expanding the size and diversity of the training set. This technique
involves generating perturbed variants of existing images through controlled
transformations. Common augmentation operations include:

    - Horizontal and vertical flipping
    - Rotation
    - Shearing
    - Zooming
    - Cropping
    - Color jittering

These transformations help improve the model’s robustness and reduce sensitivity
to variations in real-world data.

**Dataset**
This example uses the *Caltech-101* dataset, available from:
    https://data.caltech.edu/records/mzrjq-6wc02

It is assumed that the dataset has been downloaded and extracted to the following directory:
    ~/.keras/datasets/caltech_101_ObjectCategories/caltech-101/

**Reference**
Li, F.-F., Andreeto, M., Ranzato, M., & Perona, P. (2022). *Caltech 101 (1.0)* [Data set].
CaltechDATA. https://doi.org/10.22002/D1.20086

**History**
    - 2025-11-01 | I Putu Mahendra Wijaya | Initial creation
"""

from typing import Tuple, List, Any

import os
import tarfile
import pathlib
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Create aliases
AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_network(
    width: int,
    height: int,
    depth: int,
    classes: int
) -> Model:
    """
    Build a compact VGG-style convolutional neural network for image classification.

    This function constructs a lightweight variant of the VGG architecture using
    stacked convolutional, batch normalization, pooling, and dropout layers.
    The network is designed for image classification tasks with a modest number of
    input samples or limited computational resources.

    Architecture Overview:
        - Multiple convolutional blocks with ReLU activation and batch normalization
        - Max-pooling for spatial downsampling
        - Dropout layers for regularization
        - Fully connected (dense) layers for classification
        - Softmax activation at the output for multi-class prediction

    :param width:
    :type width: int

    :param height:
    :type height: int

    :param depth:
    :type depth: int

    :param classes:
    :type classes: int

    :return: VGG network
    :rtype: Model
    """

    input_layer: tf.Tensor = Input(shape=(width, height, depth))

    # First block of two convolutional layers with 32 filters each
    X: tf.Tensor = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
    )(input_layer)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
    )(X)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
    )(X)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)

    # Second block of two convolutional layers with 64 filters each
    X = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
    )(X)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
    )(X)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)

    # Last part of the architecture with a series of fully connected layers
    X = Flatten()(X)
    X = Dense(units=512)(X)
    X = ReLU()(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dropout(0.5)(X)
    X = Dense(units=classes)(X)
    output_layer: tf.Tensor = Softmax()(X)

    return Model(
        inputs=input_layer,
        outputs=output_layer
    )



def plot_model_history(
        model_history: tf.keras.callbacks.History,
        metric: str,
        plot_name: str = "Model Performance",
) -> None:
    """
    Plot and save training performance curves from a Keras model history.

    This function visualizes the progression of a specified training metric
    (e.g., accuracy, loss, precision) across epochs, based on the history
    object returned by `model.fit()`. It generates a line plot using TensorFlow
    Docs' `HistoryPlotter` utility and saves the resulting figure as a PNG file.

    Notes:
        - The plot is saved in the current working directory.
        - The y-axis is constrained to the range [0, 1] for clarity.
        - To adjust visualization styles, modify the Matplotlib style or axes limits.

    :param model_history:
    :type model_history: tf.keras.callbacks.History

    :param metric: the metric to plot
    :type metric: str

    :param plot_name: the name of the plot
    :type plot_name: str

    :return: None
    """
    plt.style.use("ggplot")
    plotter: tfdocs.plots.HistoryPlotter = tfdocs.plots.HistoryPlotter()
    plotter.plot(
        {"Model": model_history},
        metric=metric,
    )

    plt.title(f"{plot_name} - {metric.upper()}")
    plt.ylim([0, 1])

    plt.savefig(f"{plot_name}_{metric.upper()}.png")
    plt.close()


def load_image_and_label(
        image_path: str,
        target_size: Tuple[int, int] = (64, 64),
) -> Tuple:

    """
    Load an image from disk and return a resized tensor with a one-hot label.

    The function reads a JPEG image, decodes it to 3 channels (RGB), converts
    it to ``tf.float32`` in the ``[0, 1]`` range, and resizes it to
    ``target_size``. The class label is inferred from the parent directory name
    of ``image_path`` and converted to a one-hot vector against ``CLASSES``.

    Assumptions
    -----------
    - The path layout follows ``.../<class_name>/<filename>.jpg``.
    - ``CLASSES`` is an ordered sequence (e.g., ``list``/``tf.constant``) of
      valid class names, where ``len(CLASSES)`` equals the number of classes.
    - ``target_size`` is specified as ``(height, width)`` to match
      :func:`tf.image.resize`.

    :param image_path: Absolute or relative path to the input image file.
    :type image_path: str
    :param target_size: Target spatial size ``(height, width)`` in pixels used for
                        resizing the image. Defaults to ``(64, 64)``.
    :type target_size: Tuple[int, int]

    :returns: A tuple ``(image, label)`` where:
              - ``image`` is a tensor of shape ``(height, width, 3)`` with dtype
                ``tf.float32`` in the range ``[0, 1]``.
              - ``label`` is a one-hot tensor of shape ``(len(CLASSES),)`` with
                dtype ``tf.float32``.
    :rtype: Tuple[tf.Tensor, tf.Tensor]

    .. note::
       If the inferred class name is **not** present in ``CLASSES``, the returned
       one-hot vector will be all zeros. Consider validating labels upstream if
       this is undesirable.

    """

    img: tf.Tensor = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, np.float32)
    img = tf.image.resize(img, size=target_size)

    label: tf.Tensor = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES) # one-hot encoding
    label = tf.dtypes.cast(label, tf.float32)

    return img, label


def augment(
        original_image: tf.Tensor,
        label: tf.Tensor,
) -> Tuple:
    """
     Apply random data augmentation to an image tensor while preserving its label.

     This function performs a sequence of common image augmentation operations to
     increase dataset variability and improve model generalization. The augmentations
     include random cropping, horizontal flipping, and brightness adjustment.
     The image is first padded or cropped to a fixed size before applying the random
     transformations.

     :param original_image: The original input image tensor with shape ``(H, W, 3)``
                            and dtype ``tf.float32`` in the range ``[0, 1]``.
     :type original_image: tf.Tensor
     :param label: The one-hot encoded class label corresponding to the image.
     :type label: tf.Tensor

     :returns: A tuple ``(augmented_image, label)``, where:
               - ``augmented_image`` is a randomly transformed version of the input image
                 with shape ``(64, 64, 3)``.
               - ``label`` is the same label tensor passed as input.
     :rtype: Tuple[tf.Tensor, tf.Tensor]


     .. note::
        The augmentation sequence includes:
            1. Padding or cropping the image to ``74×74`` pixels.
            2. Random cropping to ``64×64``.
            3. Random horizontal flipping.
            4. Random brightness variation up to ±0.2 delta.

        These operations are applied independently per image during each call,
        introducing stochastic variation to the dataset.
    """
    img: tf.Tensor = tf.image.resize_with_crop_or_pad(
        original_image,
        target_height=74,
        target_width=74,
    )
    img = tf.image.random_crop(img, size=(64, 64, 3))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)

    return img, label


def show_augmented_grid(
    image_paths: List[str],
    n: int = 25,
    rows: int = 5,
    cols: int = 5,
    title: str = "Training Augmentations (5×5)"
) -> None:
    """
    Sample `n` images, apply `augment`, and display/save a rows×cols grid.

    Saves a PNG named `augmented_grid_5x5.png` in the CWD.
    """
    # Build a tiny dataset from paths → (image,label) → (augmented_image,label)
    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)
        .map(load_image_and_label, num_parallel_calls=AUTOTUNE)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .take(n)
        .batch(n)
    )

    imgs, labels = next(iter(ds))  # imgs: (n, 64, 64, 3), labels: (n, C)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    fig.suptitle(title, fontsize=12)

    for ax, img, lab in zip(axes.flat, imgs, labels):
        ax.imshow(img.numpy())  # already in [0,1] float32
        ax.set_axis_off()
        # Optional: show class name above each tile
        cls_idx = int(tf.argmax(lab).numpy())
        ax.set_title(str(CLASSES[cls_idx]), fontsize=8)

    plt.tight_layout()
    plt.savefig("augmented_grid_5x5.png", dpi=150)
    plt.close()


def prepare_dataset(
        data_pattern: str
) -> tf.data.Dataset:
    """
    Prepare a TensorFlow dataset pipeline for loading and preprocessing images.

    This function constructs a :class:`tf.data.Dataset` that loads image paths
    matching the given pattern, applies the :func:`load_image_and_label` function
    to each image for preprocessing, and enables parallel mapping for performance.

    :param data_pattern: A file pattern or list of image paths to include in the dataset.
                         For example, ``"~/dataset/train/*/*.jpg"``.
    :type data_pattern: str

    :returns: A TensorFlow dataset where each element is a tuple ``(image, label)``,
              where:
                - ``image`` is a preprocessed tensor of shape ``(H, W, 3)`` with dtype
                  ``tf.float32`` in the range ``[0, 1]``.
                - ``label`` is a one-hot encoded tensor representing the image class.
    :rtype: tf.data.Dataset
    """
    return (
        tf.data.Dataset
        .from_tensor_slices(data_pattern)
        .map(load_image_and_label, num_parallel_calls=AUTOTUNE)
    )


if __name__ == "__main__":

    SEED: int = 42
    np.random.seed(SEED)

    tar_caltech_path: Path = (
        pathlib.Path.home() / ".keras" / "datasets" / "caltech_101_ObjectCategories" / "caltech-101" / "101_ObjectCategories.tar.gz"
    )

    base_path: Path = tar_caltech_path.parent / "101_ObjectCategories"

    # Only extract the dataset if it hasn't been extracted yet
    if not base_path.exists():
        tar: tarfile.TarFile = tarfile.open(tar_caltech_path)
        tar.extractall(path=tar_caltech_path.parent)
        tar.close()

    image_pattern: str = str(base_path / "*" / "*.jpg")
    image_paths: List[str] = [*glob(image_pattern)]

    # We want to exclude the "BACKGROUND_Google" class from the dataset
    image_paths = [
        each_path for each_path in image_paths
        if each_path.split(os.path.sep)[-2] != "BACKGROUND_Google"
    ]

    print(f"Total number of images: {len(image_paths)}")


    CLASSES: np.ndarray = np.unique(
        [
            each_path.split(os.path.sep)[-2]
            for each_path in image_paths
        ]
    )

    (
        train_paths,
        test_paths
    ) = train_test_split(
        image_paths,
        test_size=0.2,
        random_state=SEED,
    )

    # Prepare the training and test datasets without augmentation
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 1024

    train_dataset: tf.data.Dataset = (
        prepare_dataset(train_paths)
        .batch(batch_size=BATCH_SIZE)
        .shuffle(buffer_size=BUFFER_SIZE)
        .prefetch(buffer_size=BUFFER_SIZE)
    )

    test_dataset: tf.data.Dataset = (
        prepare_dataset(test_paths)
        .batch(batch_size=BATCH_SIZE)
        .prefetch(buffer_size=BUFFER_SIZE)
    )

    # Instantiate, compile, train, and evaluate the model
    EPOCHS: int = 40
    model: Model = build_network(
        width=64,
        height=64,
        depth=3,
        classes=len(CLASSES),
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history: tf.keras.callbacks.History = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
    )

    result: tf.Tensor = model.evaluate(test_dataset)
    print(f" Test Accuracy: {result[1]}")
    plot_model_history(history, metric="accuracy", plot_name="Normal | Without Augmentation")

    # Prepare the training and test datasets WITH augmentation

    # 🔎 Visualize 25 augmented samples in a 5×5 panel (saved as augmented_grid_5x5.png)
    show_augmented_grid(
        image_paths=train_paths,
        n=25,
        rows=5,
        cols=5,
        title="Training Augmentations (5×5)"
    )

    train_dataset_aug: tf.data.Dataset = (
        prepare_dataset(train_paths)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size=BATCH_SIZE)
        .shuffle(buffer_size=BUFFER_SIZE)
        .prefetch(buffer_size=BUFFER_SIZE)
    )

    test_dataset_aug: tf.data.Dataset = (
        prepare_dataset(test_paths)
        .batch(batch_size=BATCH_SIZE)
        .prefetch(buffer_size=BUFFER_SIZE)
    )

    model_aug: Model = build_network(
        width=64,
        height=64,
        depth=3,
        classes=len(CLASSES),
    )

    model_aug.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history_aug: tf.keras.callbacks.History = model_aug.fit(
        train_dataset_aug,
        epochs=EPOCHS,
        validation_data=test_dataset_aug,
    )

    result_aug: tf.Tensor = model_aug.evaluate(test_dataset_aug)
    print(f" Test Accuracy: {result_aug[1]}")
    plot_model_history(history_aug, metric="accuracy", plot_name="Augmented | With Augmentation")