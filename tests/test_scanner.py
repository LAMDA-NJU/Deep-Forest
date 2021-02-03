import torch
import numpy as np
from deepforest.scanner import ImageScanner


def _prepare_data():
    raw_mat = torch.from_numpy(np.arange(1, 19, 1)).view(1, 2, 3, 3)

    padding_mat = raw_mat.new_zeros((1, 2, 5, 5))
    padding_mat[:, :, 1:4, 1:4] = raw_mat

    results = raw_mat.new_zeros((9, 2, 3, 3))
    for i in range(3):
        for j in range(3):
            sub_block = padding_mat[:, :, i:i+3, j:j+3]
            results[i*3+j, :, :, :] = sub_block.squeeze()
    return raw_mat, results


def test_pytorch():
    scanner = ImageScanner(3, 1, 1, "torch")

    X, results = _prepare_data()
    output = scanner.fit_transform(X)
    assert(torch.equal(output, results))
