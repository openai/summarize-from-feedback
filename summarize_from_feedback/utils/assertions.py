import numpy as np
import numpy.testing as npt
import torch

from summarize_from_feedback.utils.torch_utils import to_numpy


def _convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return to_numpy(x)
    if isinstance(x, np.ndarray):
        return x
    return None


def _maybe_convert_to_numpy(x):
    new_x = _convert_to_numpy(x)
    if new_x is None:
        return x
    return new_x


def assert_eq(x, y, err_msg=""):
    x = _maybe_convert_to_numpy(x)
    y = _maybe_convert_to_numpy(y)

    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()

    assert x == y, f"{err_msg or 'assert_eq fail'}: {x} != {y}"


def assert_allclose(x, y, err_msg="", rtol=1e-7, atol=0, equal_nan=True):
    x = _maybe_convert_to_numpy(x)
    y = _maybe_convert_to_numpy(y)
    npt.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg)


def assert_shape_eq(x, expected_shape, err_msg=""):
    if isinstance(x, torch.Tensor):
        shape = x.size()
    elif isinstance(x, np.ndarray):
        shape = x.shape
    else:
        raise Exception(f"Not sure how to take shape for type {type(x)}: {x}")

    assert shape == tuple(
        expected_shape
    ), f"{err_msg or 'assert_shape_eq fail'}: {shape} != {tuple(expected_shape)}"
