import numpy as np
import torch
import torch.nn.functional as F


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, float):
        return np.array(x)
    raise ValueError(f"Unexpected type {type(x)}")


def tensors_to_device(data, device):
    if data is None:
        return None
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: tensors_to_device(v, device) for k, v in data.items()}
    else:
        raise ValueError(f"Unsupported type: {type(data)}")


def nans(shape, dtype, device):
    return torch.ones(shape, dtype=dtype, device=device) * float("nan")


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(
        row_len, dtype=dtype, device=bools.device
    )
    return torch.min(zero_or_index, dim=-1).values


def gather_one(x, indices, *, dim):
    """
    Gather with only one element along the gathered dimension
    """
    return torch.gather(x, dim=dim, index=indices.unsqueeze(dim)).squeeze(dim)


def label_logprobs(*, logits, labels):
    """cross-entropy for arbitrary shapes"""
    assert logits.shape[:-1] == labels.shape, f"{logits.shape}[:-1] != {labels.shape}"
    flat_logits = logits.contiguous().view([-1, logits.size(-1)])
    flat_labels = labels.contiguous().view([-1])
    flat_logprobs = -F.cross_entropy(input=flat_logits, target=flat_labels, reduction="none")
    return flat_logprobs.view(labels.shape)
