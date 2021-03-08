import numpy as np
from numpy.testing import assert_array_equal

from deepforest.preprocessing import ImageScanner


def _prepare_data(b, c, h, w):
    total_items = b * c * h * w
    raw_mat = np.arange(1, total_items + 1, 1, dtype=np.uint8)
    raw_mat = raw_mat.reshape(b, c, h, w)
    y = np.array([0])

    padding_mat = np.zeros((b, c, h + 2, w + 2), dtype=np.uint8)
    padding_mat[:, :, 1 : h + 1, 1 : w + 1] = raw_mat  # noqa

    results = np.zeros((b * h * w, c, 3, 3), dtype=np.uint8)
    for idx_b in range(b):
        for idx_h in range(h):
            for idx_w in range(w):
                sub_block = padding_mat[
                    idx_b, :, idx_h : idx_h + 3, idx_w : idx_w + 3  # noqa
                ]  # noqa
                results[
                    idx_b * (h * w) + idx_h * w + idx_w, :, :, :
                ] = sub_block  # noqa

    return raw_mat, results, y


def _test_numpy_backend_with_param(b, c, h, w):
    scanner = ImageScanner(3, 1, 1, "torch")
    X, expected, y = _prepare_data(b, c, h, w)
    expected = expected.reshape(-1, c * 3 * 3)
    output, _ = scanner.fit_transform(X, y)
    assert_array_equal(output, expected)


def test_numpy_backend():
    _test_numpy_backend_with_param(1, 2, 3, 3)
    _test_numpy_backend_with_param(3, 3, 9, 9)


def _test_pytorch_backend_with_param(b, c, h, w):
    scanner = ImageScanner(3, 1, 1, "torch")
    X, expected, y = _prepare_data(b, c, h, w)
    expected = expected.reshape(-1, c * 3 * 3)
    output, _ = scanner.fit_transform(X, y)
    assert_array_equal(output, expected)


def test_pytorch_backend():
    _test_pytorch_backend_with_param(1, 2, 3, 3)
    _test_pytorch_backend_with_param(3, 3, 9, 9)


def test_horizontal_same():
    """Make sure the numpy and torch backend have the same output."""
    numpy_scanner = ImageScanner(3, 1, 1, "numpy")
    X, _, y = _prepare_data(1, 2, 3, 3)
    numpy_output, _ = numpy_scanner.fit_transform(X, y)

    torch_scanner = ImageScanner(3, 1, 1, "torch")
    torch_output, _ = torch_scanner.fit_transform(X, y)

    assert_array_equal(numpy_output, torch_output)
