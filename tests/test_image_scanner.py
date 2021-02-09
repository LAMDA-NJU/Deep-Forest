import numpy as np
from numpy.testing import assert_array_equal

from deepforest.preprocessing import ImageScanner


def _prepare_data():
    raw_mat = np.arange(1, 19, 1, dtype=np.uint8).reshape(1, 2, 3, 3)
    y = np.array([0])

    padding_mat = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    padding_mat[:, :, 1:4, 1:4] = raw_mat

    results = np.zeros((9, 2, 3, 3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            sub_block = padding_mat[:, :, i : i + 3, j : j + 3]  # noqa
            results[i * 3 + j, :, :, :] = sub_block.squeeze()

    return raw_mat, results, y


def test_numpy_backend():
    scanner = ImageScanner(3, 1, 1, "numpy")
    X, expected, y = _prepare_data()
    expected = expected.reshape(9, 18)
    output, _ = scanner.fit_transform(X, y)
    assert_array_equal(output, expected)


def test_pytorch_backend():
    scanner = ImageScanner(3, 1, 1, "torch")
    X, expected, y = _prepare_data()
    expected = expected.reshape(9, 18)
    output, _ = scanner.fit_transform(X, y)
    assert_array_equal(output, expected)


def test_horizontal_same():
    """Make sure the numpy and torch backend have the same output."""
    numpy_scanner = ImageScanner(3, 1, 1, "numpy")
    X, _, y = _prepare_data()
    numpy_output, _ = numpy_scanner.fit_transform(X, y)

    torch_scanner = ImageScanner(3, 1, 1, "torch")
    torch_output, _ = torch_scanner.fit_transform(X, y)

    assert_array_equal(numpy_output, torch_output)
