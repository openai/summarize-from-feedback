import math

import torch
from torch import nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


ACT_FNS = {
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu": gelu,
    "quick_gelu": quick_gelu,
    "gelu2": quick_gelu,
}


def NormalParameter(n_in, n_out, init_scale=1.0):
    """Parameter with random normal initialization"""
    w = torch.empty(n_in, n_out)
    nn.init.normal_(w, std=0.02 * init_scale)
    return nn.Parameter(w)


class Conv1D(nn.Module):
    def __init__(self, n_in, n_out, zero_out=False, bias=True, init_scale=1.0):
        super(Conv1D, self).__init__()

        assert not zero_out, "This value is deprecated"

        self.n_in = n_in
        self.n_out = n_out
        self.weight = NormalParameter(n_in, n_out, init_scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out))
        else:
            self.bias = None

    def forward(self, x):
        size_out = (*x.size()[:-1], self.n_out)
        if self.bias is not None:
            x = torch.addmm(
                self.bias.type_as(x), x.contiguous().view(-1, x.size(-1)), self.weight.type_as(x)
            )
        else:
            x = torch.mm(x.contiguous().view(-1, x.size(-1)), self.weight.type_as(x))
        x = x.view(*size_out)
        return x
