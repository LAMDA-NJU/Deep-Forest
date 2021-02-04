import os
import pytest
import numpy as np

from deepforest import _io as io


open_buffer = io.Buffer(
    use_buffer=True,
    buffer_dir="./",
    store_est=True,
    store_pred=True,
    store_data=True,
)


close_buffer = io.Buffer(use_buffer=False)

X = np.zeros((42, 42), dtype=np.uint8)


def test_buffer_name():
    name = open_buffer.name
    assert isinstance(name, str)

    name = close_buffer.name
    assert name is None


def test_store_data_close_buffer():
    """When `store_data` is False, the buffer directly returns the array."""
    ret = close_buffer.cache_data(0, X)
    assert isinstance(ret, np.ndarray)


def test_store_data_open_buffer():
    """
    When `store_data` is True, the buffer returns the memmap object of the
    dumped array.
    """
    layer_idx = 0
    ret = open_buffer.cache_data(layer_idx, X, is_training_data=True)
    assert isinstance(ret, np.memmap)
    assert os.path.exists(
        os.path.join(
            open_buffer.data_dir_, "joblib_train_{}.mmap".format(layer_idx)
        )
    )

    ret = open_buffer.cache_data(layer_idx, X, is_training_data=False)
    assert isinstance(ret, np.memmap)
    assert os.path.exists(
        os.path.join(
            open_buffer.data_dir_, "joblib_test_{}.mmap".format(layer_idx)
        )
    )


def test_load_estimator_missing():
    err_msg = "Missing estimator in the path: unknown.est."
    with pytest.raises(FileNotFoundError, match=err_msg):
        open_buffer.load_estimator("unknown.est")


def test_load_predictor_missing():
    err_msg = "Missing predictor in the path: unknown.est."
    with pytest.raises(FileNotFoundError, match=err_msg):
        open_buffer.load_predictor("unknown.est")
