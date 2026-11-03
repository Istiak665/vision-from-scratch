import numpy as np

from scratchvision.data.cifar10 import load_cifar10
from scratchvision.data.cifar10 import load_class_names

X_train, y_train, X_test, y_test = load_cifar10()


def test_training_shapes():
    assert X_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000,)


def test_test_shapes():
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000,)


def test_pixel_range():
    assert X_train.dtype == np.uint8
    assert X_train.min() >= 0
    assert X_train.max() <= 255


def test_class_balance():
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)

    assert np.all(train_counts == 5000)
    assert np.all(test_counts == 1000)


def test_class_names():
    class_names = load_class_names()

    assert len(class_names) == 10
    assert class_names[0] == "airplane"
    assert class_names[9] == "truck"