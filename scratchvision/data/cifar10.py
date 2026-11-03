from pathlib import Path
import pickle

import numpy as np

# cifar10.py -> data -> scratchvision -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_FOLDER = PROJECT_ROOT / "datasets" / "cifar-10-batches-py"


def load_batch(batch_path):
    """Load one CIFAR-10 batch file.

    Returns:
        images: uint8 array of shape (N, 32, 32, 3)
        labels: int64 array of shape (N,)
    """
    # "rb" means read binary
    with open(batch_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    raw_images = batch[b"data"]
    labels = np.array(batch[b"labels"], dtype=np.int64)

    number_of_images = raw_images.shape[0]

    # Shape: (N, 3072) -> (N, 3, 32, 32)
    images = raw_images.reshape(number_of_images, 3, 32, 32)

    # Shape: (N, 3, 32, 32) -> (N, 32, 32, 3)
    images = images.transpose(0, 2, 3, 1)

    return images, labels


def load_class_names(data_folder=DEFAULT_DATA_FOLDER):
    """Return the 10 class names as a list of strings."""
    meta_path = Path(data_folder) / "batches.meta"

    with open(meta_path, "rb") as file:
        meta = pickle.load(file, encoding="bytes")

    class_names = []

    for name in meta[b"label_names"]:
        class_names.append(name.decode("utf-8"))

    return class_names


def load_cifar10(data_folder=DEFAULT_DATA_FOLDER):
    """Load the full CIFAR-10 dataset.

    Returns:
        X_train: uint8 array of shape (50000, 32, 32, 3)
        y_train: int64 array of shape (50000,)
        X_test: uint8 array of shape (10000, 32, 32, 3)
        y_test: int64 array of shape (10000,)
    """
    data_folder = Path(data_folder)

    image_batches = []
    label_batches = []

    # Load data_batch_1 to data_batch_5
    for batch_number in range(1, 6):
        batch_path = data_folder / f"data_batch_{batch_number}"
        images, labels = load_batch(batch_path)
        image_batches.append(images)
        label_batches.append(labels)

    # Join the 5 batches along the first axis
    X_train = np.concatenate(image_batches, axis=0)
    y_train = np.concatenate(label_batches, axis=0)

    # Load the test set separately
    X_test, y_test = load_batch(data_folder / "test_batch")

    return X_train, y_train, X_test, y_test