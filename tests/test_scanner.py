import numpy as np
from deepforest.scanner import ImageScanner


def _prepare_data():
    raw_mat = np.arange(1, 19, 1, dtype=np.uint8).reshape(1, 2, 3, 3)

    padding_mat = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    padding_mat[:, :, 1:4, 1:4] = raw_mat

    results = np.zeros((9, 2, 3, 3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            sub_block = padding_mat[:, :, i:i+3, j:j+3]
            results[i*3+j, :, :, :] = sub_block.squeeze()
    return raw_mat, results


def test_pytorch_channels_first():
    scanner = ImageScanner(3, 1, 1, "torch", channels_first=True)

    X, results = _prepare_data()
    results = results.reshape(9, 18)
    output = scanner.fit_transform(X)
    assert (output == results).all()


def test_pytorch_channels_last():
    scanner = ImageScanner(3, 1, 1, "torch", channels_first=False)

    X, results = _prepare_data()
    X = X.transpose((0, 2, 3, 1))
    results = results.transpose((0, 2, 3, 1)).reshape(9, 18)
    output = scanner.fit_transform(X)
    assert (output == results).all()
