import os
import shutil
import pytest

from deepforest import _io as io


save_dir = "./tmp"


def test_mkdir_normal():
    io.model_mkdir(save_dir)

    assert os.path.isdir(save_dir) is True
    assert os.path.isdir(os.path.join(save_dir, "estimator")) is True

    shutil.rmtree(save_dir)


def test_mkdir_already_exist():
    os.mkdir(save_dir)

    err_msg = "The directory to be created already exists ./tmp."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_mkdir(save_dir)

    shutil.rmtree(save_dir)


def test_model_saveobj_not_exist():
    err_msg = (
        "Cannot find the target directory: ./tmp." " Please create it first."
    )
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_saveobj(save_dir, "param", None)


def test_model_saveobj_param_binner_invalid_data_type():
    io.model_mkdir(save_dir)

    obj = ["binner_1", "binner_2"]
    err_msg = "param to be saved should be in the form of dict."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_saveobj(save_dir, "param", obj)

    shutil.rmtree(save_dir)


def test_model_saveobj_layer_invalid_data_type():
    io.model_mkdir(save_dir)

    obj = ["layer_1", "layer_2"]
    err_msg = "The layer to be saved should be in the form of dict."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_saveobj(save_dir, "layer", obj)

    shutil.rmtree(save_dir)


def test_model_saveobj_layer_missing_estimator_dir():
    io.model_mkdir(save_dir)
    estimator_dir = os.path.join(save_dir, "estimator")
    shutil.rmtree(estimator_dir)

    obj = {1: "layer_1"}
    with pytest.raises(RuntimeError) as execinfo:
        io.model_saveobj(save_dir, "layer", obj)
    assert "Cannot find the target directory" in str(execinfo.value)

    shutil.rmtree(save_dir)


def test_model_saveobj_type_unknown():
    io.model_mkdir(save_dir)

    err_msg = "Unknown object type: unknown."
    with pytest.raises(ValueError, match=err_msg):
        io.model_saveobj(save_dir, "unknown", None)

    shutil.rmtree(save_dir)


def test_model_loadobj_missing_dir():
    err_msg = "Cannot find the target directory: ./tmp."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_loadobj(save_dir, "param")


def test_model_loadobj_layer_missing_param():
    io.model_mkdir(save_dir)

    err_msg = "Loading layers requires the dict from `param.pkl`."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_loadobj(save_dir, "layer", None)

    shutil.rmtree(save_dir)


def test_model_loadobj_predictor_missing_param():
    io.model_mkdir(save_dir)

    err_msg = "Loading the predictor requires the dict from `param.pkl`."
    with pytest.raises(RuntimeError, match=err_msg):
        io.model_loadobj(save_dir, "predictor", None)

    shutil.rmtree(save_dir)


def test_model_loadobj_type_unknown():
    io.model_mkdir(save_dir)

    err_msg = "Unknown object type: unknown."
    with pytest.raises(ValueError, match=err_msg):
        io.model_loadobj(save_dir, "unknown")

    shutil.rmtree(save_dir)
