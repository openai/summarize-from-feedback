import numpy as np
import pytest
import torch

from summarize_from_feedback.utils.assertions import assert_eq, assert_allclose
from summarize_from_feedback.utils.torch_utils import first_true_indices, label_logprobs

only_if_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=only_if_cuda)])
def test_first_true_indices(device):
    bools = torch.tensor(
        [
            [False, True, True],
            [True, True, False],
            [True, False, True],
            [False, False, False],
            [True, False, False],
            [False, False, True],
        ],
        device=device,
    )
    want = torch.tensor([1, 0, 0, 3, 0, 2], device=device)
    assert_eq(first_true_indices(bools), want)

    for bool_row, row_want in zip(bools, want):
        assert_eq(first_true_indices(bool_row), row_want)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=only_if_cuda)])
def test_label_logprobs(device):
    assert_allclose(
        label_logprobs(logits=torch.tensor([0.0, 0.0]), labels=torch.tensor(1)), np.log(1 / 2)
    )
    assert_allclose(
        label_logprobs(
            logits=torch.tensor([[2.0, 2.0, 2.0], [1.0, 0.0, -np.inf]]), labels=torch.tensor([2, 0])
        ),
        [np.log(1 / 3), np.log(np.e / (np.e + 1))],
    )
    assert_allclose(
        label_logprobs(
            logits=torch.tensor([[[2.0, 2.0]], [[1.0, 0.0]]]), labels=torch.tensor([[0], [1]])
        ),
        [[np.log(1 / 2)], [np.log(1 / (np.e + 1))]],
    )
