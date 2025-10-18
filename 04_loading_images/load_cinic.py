"""
This script demonstrates how to load the CINIC-10 dataset using the Keras API.
It provides a simple example of how to load and preprocess the dataset for use in image classification tasks.

History:
- 2025 October 17 | I Putu Mahendra Wijaya | Initial creation
- 2025 October 18 | I Putu Mahendra Wijaya | Updated to use tf.data.Dataset API

"""
from typing import List, Any
import sys
import glob
import os
import tarfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import get_file
from PIL import ImageFile
import numpy as np

def load_images_using_keras_api():
    """
    Function to demonstrate how to load the CINIC-10 dataset using the conventional keras API

    :param: None
    :return: None
    """
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

    # Load all image paths and print the number of images found
    img_patterns: str = os.path.sep.join([data_dir, "**", "*.png"])
    lst_img_paths: List = glob.glob(img_patterns, recursive=True)
    print(f"Found {len(lst_img_paths)} images in {data_dir}")

    # Load single image and print its metadata
    sample_img: ImageFile = load_img(lst_img_paths[0])
    print(
        f"""
        Sample image metadata:
        ------------------------------------------------
        
        Image Type: {type(sample_img)}
        Image Format: {sample_img.format}
        Image Mode: {sample_img.mode}
        Image Size: {sample_img.size}
        """
    )

    # Convert image to numpy array
    sample_img_arr: np.ndarray = img_to_array(sample_img)
    print(
        f"""
        After converting to numpy array:
        --------------------------------------------
        
        Image Type: {type(sample_img_arr)}
        Image Array Shape: {sample_img_arr.shape}
        """
    )

    # Display an image using matplotlib
    plt.imshow(sample_img_arr / 255.0)
    plt.axis("off")
    plt.title(
        "Sample Image",
        fontsize=12
    )
    plt.show()

    # Load a batch of images using `ImageDataGenerator`
    datagen: ImageDataGenerator = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1.0 / 255.0
    )
    batch_size: int = 10
    img_gen: ImageDataGenerator = datagen.flow_from_directory(
        directory=data_dir,
        batch_size=batch_size,
    )

    for each_batch, _ in img_gen:
        print(f"Batch shape: {each_batch.shape}")
        plt.figure(figsize=(10, 10))

        for each_idx, img in enumerate(each_batch, start=1):
            ax = plt.subplot(5, 5, each_idx)
            plt.imshow(img)
            plt.axis("off")

        plt.show()
        break


def load_images_using_tf_dataset():
    """
    Function to demonstrate how to load the CINIC-10 dataset using the tf.data.Dataset API
    
    the td.data.Dataset API is a high-level API for building input pipelines. 
    
    Its functional-style interface, as well as its high-level of optimization, makes it better 
    than conventional Keras API for large projects, where efficiency and resource consumption matters 
    
    :param: None
    :return: None
    """
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

    # Load all image paths and print the number of images found using tf.data.Dataset API
    img_patterns: str = os.path.sep.join([data_dir, "*/*/*.png"])
    img_dataset: DatasetV2 = tf.data.Dataset.list_files(img_patterns)

    # Take a single path from the dataset
    for each_img_path in img_dataset.take(1):
        _sample_path: np.ndarray = each_img_path.numpy()

    sample_img: Any  = tf.io.read_file(_sample_path)

    # decode the image to machine-readable numpy array
    sample_img_arr: Any = tf.image.decode_png(sample_img, channels=3)
    sample_img_arr = sample_img_arr.numpy()

    # Display an image using matplotlib
    plt.imshow(sample_img_arr / 255.0)
    plt.axis("off")
    plt.title(
        "Sample Image",
        fontsize=12
    )
    plt.show()

    # Load a batch of images using tf.io.read_file and tf.image.decode_png
    plt.figure(figsize=(10, 10))
    for each_idx, img_path in enumerate(img_dataset.take(10), start=1):
        img_rf: Any = tf.io.read_file(img_path)
        img_dcd: Any = tf.image.decode_png(img_rf, channels=3)
        img_arr: np.ndarray = tf.image.convert_image_dtype(img_dcd, tf.float32)
        plt.subplot(5, 5, each_idx)
        plt.imshow(img_arr.numpy())
        plt.axis("off")

    plt.show()
    plt.close()


if __name__ == "__main__":

    if len(sys.argv) > 1:

        func_name: str = sys.argv[1]

        if func_name == "load_images_using_keras_api":
            load_images_using_keras_api()

        if func_name == "load_images_using_tf_dataset":
            load_images_using_tf_dataset()

        else:
            print(f"Function {func_name} not found.")

    else:
        print("Usage: python load_cinic.py <function_name>")
        print(
        """
        Available functions:
        * load_images_using_keras_api
        * load_images_using_tf_dataset
        """
        )