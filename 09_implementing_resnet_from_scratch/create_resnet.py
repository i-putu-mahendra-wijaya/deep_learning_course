"""
ResNet implementation from scratch.

In this script, we will delve into the implementation of Residual Networks (ResNets) and explore their
architecture, training, and evaluation. ResNet is one of the ground-breaking advances in deep learning
It allows for the training of very deep neural networks by introducing skip connections,
which help mitigate the vanishing gradient problem and enable the training of deeper models.

There are variants of ResNet that have more than 100 layers, without any loss of performance.
ResNet-152, for example, has 152 layers and has been shown to achieve state-of-the-art performance on
image classification tasks.

Interested students can refer to the following papers for more theoretical foundation behind ResNet:
https://arxiv.org/pdf/1512.03385

In this script, we will implement a ResNet from scratch and train it on
the CINIC-10 dataset. We will use the Keras API to build the model.

History:
- 2025 October 26 | I Putu Mahendra Wijaya | Initial creation

"""

from typing import Any, Tuple, Union, List, Optional
import os
import numpy as np
import tarfile
import tensorflow as tf
from PIL import ImageFile
from tensorflow import RaggedTensor
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

# Defining Aliases
AUTOTUNE = tf.data.experimental.AUTOTUNE
CINIC_MEAN_RGB: np.ndarray = np.array([0.47889522, 0.47227842, 0.43047404]) # Numpy array to normalize the CINIC-10 image dataset
CINIC_10_CLASSEs: tf.Tensor = tf.constant(
    value=[
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    dtype=tf.string
)
BATCH_SIZE: int = 128
BUFFER_SIZE: int = 1024


def residual_module(
        data: tf.Tensor,
        filters: int,
        stride: Union[int, Tuple[int, int]] = 1,
        reduce: bool = False,
        reg: float = 0.0001,
        bn_eps: float = 2e-5,
        bn_momentum: float = 0.9,
) -> tf.Tensor:

    """
    Implementing a Residual Module

    This function implements a residual module, which consists of three convolutional layers.
    If reduce is set to True, we will apply a 1x1 convolution to reduce the number of channels, else
    we will use the original data as the shortcut.

    :param data: Input Tensor
    :type data: tf.Tensor

    :param filters: Number of output channels of the block (post-bottleneck)
    :type filters: int

    :param stride: Stride for spatial downsampling (applied to the 3x3 conv and shortcut)
    :type stride: Union[int, Tuple[int, int]]

    :param reduce: If set to True, we will apply a 1x1 convolution to reduce the number of channels
    :type reduce: bool

    :param reg: L2 regularization factor for conv kernels
    :type reg: float

    :param bn_eps: Epsilon for batch normalization
    :type bn_eps: float

    :param bn_momentum: Momentum for batch normalization
    :type bn_momentum: float

    :return: Output tensor after residual addition
    :rtype: tf.Tensor
    """

    # Normalize stride to tuple
    if isinstance(stride, int):
        stride = (stride, stride)

    # Implementing First Block
    bn_1: BatchNormalization = BatchNormalization(
        axis=-1,
        epsilon=bn_eps,
        momentum=bn_momentum,
    )(data)
    act_1: ReLU = ReLU()(bn_1)
    conv_1: Conv2D = Conv2D(
        filters=int(filters * 0.25),
        kernel_size=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(reg),
    )(act_1)

    # Implementing Second Block
    bn_2: BatchNormalization = BatchNormalization(
        axis=-1,
        epsilon=bn_eps,
        momentum=bn_momentum,
    )(conv_1)
    act_2: ReLU = ReLU()(bn_2)
    conv_2: Conv2D = Conv2D(
        filters=int(filters * 0.25),
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(reg),
    )(act_2)

    # Implementing Third Block
    bn_3: BatchNormalization = BatchNormalization(
        axis=-1,
        epsilon=bn_eps,
        momentum=bn_momentum,
    )(conv_2)
    act_3: ReLU = ReLU()(bn_3)
    conv_3: Conv2D = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(reg),
    )(act_3)

    # Shortcut / projection
    need_proj: bool = (
        reduce
        or
        stride != (1, 1)
        or
        data.shape[-1] != filters
    )
    if need_proj:
        # If reduce is True, we will apply a 1x1 convolution to reduce the number of channels
        shortcut: Conv2D = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=stride,
            use_bias=False,
            kernel_regularizer=l2(reg),
        )(act_1)
    else:
        # Else, we will use the data as shortcut
        shortcut: tf.Tensor = data


    res_module: tf.Tensor = add([conv_3, shortcut])

    return res_module


def build_resnet(
        input_shape: Tuple[int, int, int],
        classes: int,
        stages: Tuple,
        filters: Tuple,
        reg: float = 1e-3,
        bn_eps: float = 2e-5,
        bn_momentum: float = 0.9,
) -> Model:
    """
    Build a configurable ResNet architecture using residual modules.

    This function constructs a Residual Network (ResNet) model based on the given configuration of
    stages and filter sizes. Each stage consists of a stack of residual modules, where the first module
    in every stage may downsample the input feature maps (stride = (2, 2)) to progressively reduce
    spatial resolution while increasing channel depth.

    The model begins with a batch normalization and a 3×3 convolutional stem, followed by a series
    of stages defined in the `stages` list. Each stage’s depth (number of residual blocks) and
    channel width (from the `filters` list) are configurable, enabling flexible construction of
    ResNet-like architectures such as ResNet-18, ResNet-34, or custom variants.

    At the end of the network, a global average pooling layer aggregates spatial information,
    followed by a fully-connected classification head with softmax activation.

    :param input_shape: Shape of the input tensor (H, W, C).
    :type input_shape: Tuple[int, int, int]

    :param classes: Number of target classes for the final softmax output layer.
    :type classes: int

    :param stages: List specifying the number of residual modules in each stage.
    :type stages: Tuple

    :param filters: List specifying the number of filters for each corresponding stage.
    :type filters: Tuple

    :param reg: L2 regularization factor for all convolution and dense layers.
    :type reg: float

    :param bn_eps: Epsilon value for batch normalization layers.
    :type bn_eps: float

    :param bn_momentum: Momentum for batch normalization layers.
    :type bn_momentum: float

    :return: A compiled Keras Model representing the constructed ResNet.
    :rtype: tensorflow.keras.Model
    """

    # Input stem
    inputs: Input = Input(shape=input_shape)
    x: tf.Tensor = BatchNormalization(axis=-1, epsilon=bn_eps, momentum=bn_momentum)(inputs)
    x: tf.Tensor = Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(reg),
    )(x)

    # Build stages
    for stage_idx, num_blocks in enumerate(stages):
        stride: Tuple[int, int] = (1, 1) if stage_idx == 0 else (2, 2)

        # First block in each stage (possibly downsampling)
        x = residual_module(
            data=x,
            filters=filters[stage_idx + 1],
            stride=stride,
            reduce=True,
            reg=reg,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        # Remaining blocks in the same stage
        for _ in range(num_blocks - 1):
            x = residual_module(
                data=x,
                filters=filters[stage_idx + 1],
                stride=(1, 1),
                reg=reg,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )

    # Final layers
    x: tf.Tensor = BatchNormalization(axis=-1, epsilon=bn_eps, momentum=bn_momentum)(x)
    x: tf.Tensor = ReLU()(x)
    x: tf.Tensor = AveragePooling2D(pool_size=(8, 8))(x)
    x: tf.Tensor = Flatten()(x)
    x: tf.Tensor = Dense(
        units=classes,
        kernel_regularizer=l2(reg),
    )(x)
    outputs: tf.Tensor = Softmax()(x)

    return Model(inputs=inputs, outputs=outputs, name="ResNet")


def load_image_and_label(
        image_path: str,
        target_size: Tuple[int, int] = (32, 32),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess an image along with its corresponding label.

    This function:
    - reads an image from disk
    - decodes it into an RGB tensor
    - normalizes its pixel values to the [0, 1] range
    - standardizes it using the CINIC-10 dataset mean
    - resizes it to the specified target size
    - and extracts its class label from the parent directory name.

    The label is encoded as a one-hot vector based on the predefined list of CINIC-10 classes.

    :param image_path: Full filesystem path to the image file.
    :type image_path: str

    :param target_size: Target spatial resolution of the image (height, width) after resizing.
    :type target_size: Tuple[int, int]

    :return: A tuple of (preprocessed image tensor, one-hot encoded label tensor).
    :rtype: Tuple[tf.Tensor, tf.Tensor]
    """

    img: tf.Tensor = tf.io.read_file(image_path)
    img: tf.Tensor = tf.image.decode_png(img, channels=3)
    img: tf.Tensor = tf.image.convert_image_dtype(img, tf.float32)

    img: np.ndarray = img - CINIC_MEAN_RGB
    img: tf.Tensor = tf.image.resize(img, size=target_size)

    label: Union[RaggedTensor, Any] = tf.strings.split(image_path, os.path.sep)[-2]
    label: int = (label == CINIC_10_CLASSEs) # one-hot encoding
    label: tf.Tensor = tf.dtypes.cast(label, tf.float32)

    return img, label


def prepare_dataset(
        data_pattern: str,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
):
    """
    Build a TensorFlow dataset pipeline for loading and preprocessing image data.

    This function constructs a `tf.data.Dataset` pipeline that:
    1. Lists image file paths matching the given glob pattern.
    2. Loads and preprocesses each image–label pair using `load_image_and_label()`.
    3. Batches the samples for efficient GPU/TPU training.
    4. Optionally shuffles the dataset for randomized training order.
    5. Prefetches batches asynchronously to overlap preprocessing and training.

    Designed for use with CINIC-10 or similar directory-structured datasets where
    each image’s parent folder name corresponds to its class label.

    :param data_pattern: Glob pattern specifying the image file paths to include
                         (e.g., "dataset/train/*/*.png").
    :type data_pattern: str

    :param shuffle: Whether to shuffle dataset order (recommended for training).
    :type shuffle: bool

    :param batch_size: Optional override for global BATCH_SIZE constant.
    :type batch_size: Optional[int]

    :param buffer_size: Optional override for global BUFFER_SIZE constant used in shuffling.
    :type buffer_size: Optional[int]

    :return: A preprocessed, batched, and prefetched `tf.data.Dataset` ready for model training or evaluation.
    :rtype: tf.data.Dataset
    """

    batch_size_ : int = batch_size or BATCH_SIZE
    buffer_size_ : int = buffer_size or BUFFER_SIZE

    dataset: tf.data.Dataset = (
        tf.data.Dataset
        .list_files(data_pattern)
        .map(load_image_and_label, num_parallel_calls=AUTOTUNE)
        .batch(batch_size=batch_size_)
    )

    if shuffle:
        dataset: tf.data.Dataset = dataset.shuffle(buffer_size=buffer_size_)

    return dataset.prefetch(buffer_size=AUTOTUNE)


if __name__ == "__main__":
    DATASET_URL: str = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y"
    DATASET_NAME: str = "cinic-10"
    FILE_EXTENSION: str = ".tar.gz"
    FILE_NAME: str = ".".join([DATASET_NAME, FILE_EXTENSION])

    # Download the dataset
    # By default, the file will be downloaded to the cache dir (~/.keras/datasets)
    dwn_file_loc: str = get_file(
        origin=DATASET_URL,
        fname=FILE_NAME,
        extract=False
    )

    # Build path to the data directory based on the downloaded file location
    data_dir, _ = dwn_file_loc.rsplit(
        os.path.sep,
        maxsplit=1
    )
    data_dir = os.path.sep.join([data_dir, DATASET_NAME])

    # Only extract the dataset if it hasn't been extracted yet
    if not os.path.exists(data_dir):
        tar = tarfile.open(dwn_file_loc)
        tar.extractall(path=data_dir)
        tar.close()

    # Define the glob-like pattern to the train, test, and validation subsets
    train_img_patterns: str = os.path.sep.join([data_dir, "train", "*/*.png"])
    test_img_patterns: str = os.path.sep.join([data_dir, "test", "*/*.png"])
    val_img_patterns: str = os.path.sep.join([data_dir, "valid", "*/*.png"])

    # Prepare the datasets
    train_dataset: tf.data.Dataset = prepare_dataset(train_img_patterns, shuffle=True)
    test_dataset: tf.data.Dataset = prepare_dataset(test_img_patterns)
    val_dataset: tf.data.Dataset = prepare_dataset(val_img_patterns)

    # Build, compile, and train a ResNet model
    # We will save a version of the model after each epoch, because
    # this is a time-consuming process
    model: Model = build_resnet(
        input_shape=(32, 32, 3),
        classes=10,
        stages=(9, 9, 9),
        filters=(64, 64, 128, 256),
        reg=5e-3
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )

    model_checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        filepath="./model.{epoch:02d} - {val_accuracy:.2f}.keras",
        save_weights_only=False,
        monitor="val_accuracy",
    )

    EPOCHS: int = 5
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[model_checkpoint_callback],
    )

    # load back the best model
    # best_model: str = "model.38 - 0.72.hdf5" # need to check which models that is the best later
    # model: Model = tf.keras.models.load_model(best_model)
    # result: tf.Tensor = model.evaluate(test_dataset)
    # print(f"Test set accuracy: {result[1]}")
