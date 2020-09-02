from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from summarize_from_feedback.models.ops import Conv1D
from summarize_from_feedback.utils import exact_div
from summarize_from_feedback.utils.dist_utils import Comm


class Attention(nn.Module):
    """
    d_model, n_ctdx, d_attn, n_head are all standard Transformer hyperparams
    attn_dropout, resid_dropout handle dropout rates for different parameters
    init_scale: scalar which can be used to change all inits
    mp_comm: a comm for sharding
    """

    def __init__(
        self,
        d_model,
        n_ctx,
        d_attn,  # The size of the hidden states for each of k,q,v
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        zero_out=False,
        init_scale=1.0,
        mp_comm: Comm = None,
        key_bias=False,
    ):
        super(Attention, self).__init__()

        # Set up sharding. We put a subset of heads on each shard
        n_shards = 1 if mp_comm is None else mp_comm.size

        heads_per_shard = exact_div(n_head, n_shards)
        self.mp_comm = mp_comm

        d_attn_sharded = exact_div(d_attn, n_shards)

        self.q_proj = Conv1D(d_model, d_attn_sharded, init_scale=init_scale)
        self.k_proj = Conv1D(d_model, d_attn_sharded, init_scale=init_scale, bias=key_bias)
        self.v_proj = Conv1D(d_model, d_attn_sharded, init_scale=init_scale)
        self.c_proj = Conv1D(d_attn_sharded, d_model, zero_out, init_scale=init_scale)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.n_ctx = n_ctx

        self.attn_fn = AttentionFunc(
            attn_dropout_module=self.attn_dropout, heads_per_shard=heads_per_shard
        )

    def _get_past_keys_values(self, past: torch.Tensor):
        """
        figures out how much recompute to do on past
        """
        # This handles the past context
        if past is not None:
            # We are sampling, and we have some precomputed queries/keys
            #
            # Or, we evaluated a shared context and want to do forward passes on continuations
            # (not yet supported)
            assert past.dim() == 4, f"wrong past shape: {past.size()}"
            # [2, batch_n, past_ctx_n, model_dim]
            past_keys, past_values = past
            return past_keys, past_values
        else:
            return None, None

    def forward(
        self, x: torch.Tensor, hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: The new activations coming in, if this is sampling, then the activations
            will have an input_ctx_len of 1
        :param hidden_state: The past activations or past keys, values
        :return:
        activations: output of attention
        hidden_state: keys and values
        """

        query = self.q_proj(x)
        new_key = self.k_proj(x)
        new_value = self.v_proj(x)

        # returns None if no hidden_state
        past_keys, past_values = self._get_past_keys_values(hidden_state)

        if past_keys is not None:
            # key, value are from the current batch being passed through
            # the transformer
            key = torch.cat((past_keys, new_key), dim=-2)
            value = torch.cat((past_values, new_value), dim=-2)
        else:
            key = new_key
            value = new_value

        a = self.attn_fn(query, key, value)

        a = self.c_proj(a)

        if self.mp_comm is not None:
            a = self.mp_comm.all_reduce(a, "attn")

        total_ctx_len = key.size(-2)
        if total_ctx_len <= self.n_ctx:
            output_context = torch.stack([new_key, new_value])
        else:
            raise ValueError(f"hidden_state_ctx_len > self.n_ctx: {total_ctx_len} > {self.n_ctx}")

        return self.resid_dropout(a), output_context


class AttentionFunc:
    """
    After this was developed pytorch added their own implementation:
    https://pytorch.org/docs/master/nn.html#multiheadattention
    """

    def __init__(self, heads_per_shard, attn_dropout_module):
        self.attn_dropout_module = attn_dropout_module
        self.heads_per_shard = heads_per_shard

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        # query: [batch, head, n_q, d_model]
        # key: [batch, head, d_model, n_k]
        # value: [batch, head, n_k, d_model]

        # Pre-divide by fp16_stability_scale to prevent fp16 overflow
        softmax_scale = 1.0 / np.sqrt(np.sqrt(query.size(-1)))
        query = query * softmax_scale
        key = key * softmax_scale

        w = torch.matmul(query, key)
        wtype = w.dtype
        w = w.float()

        # Dense attn with autoregressive mask
        n_q = w.size(-2)
        n_k = w.size(-1)

        # NOTE: Could use apex prefix softmax to speed this up

        mask = torch.ones(n_q, n_k, device=w.device).tril(diagonal=n_k - n_q).view(1, 1, n_q, n_k)

        # We make all values where the mask==0 into -inf so that they get
        # ignored when we do our softmax
        w = w * mask + -1e9 * (1 - mask)

        w = nn.Softmax(dim=-1)(w).type(wtype)

        w = self.attn_dropout_module(w)
        a = torch.matmul(w, value)
        # a: [batch, head, n_q, d_model]
        a = self.merge_heads(a)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = (*x.size()[:-2], x.size(-2) * x.size(-1))
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = (*x.size()[:-1], self.heads_per_shard, x.size(-1) // self.heads_per_shard)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)
