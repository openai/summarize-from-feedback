import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm as _LayerNorm

from summarize_from_feedback.model_layout import ModelLayout
from summarize_from_feedback.models.attention import Attention
from summarize_from_feedback.models.ops import ACT_FNS, Conv1D
from summarize_from_feedback.utils import exact_div
from summarize_from_feedback.utils.dist_utils import create_model_parallel_comm, Comm


class Hyperparams(dict):
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


class LayerNorm(_LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class MLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        resid_dropout=0.0,
        afn="quick_gelu",
        zero_out=False,
        init_scale=1.0,
        mp_comm: Comm = None,
    ):
        super().__init__()

        n_shards = 1 if mp_comm is None else mp_comm.size
        assert d_ff % n_shards == 0
        self.mp_comm = mp_comm
        self.c_fc = Conv1D(d_model, exact_div(d_ff, n_shards), init_scale=init_scale)
        self.c_proj = Conv1D(exact_div(d_ff, n_shards), d_model, zero_out, init_scale=init_scale)
        self.act = ACT_FNS[afn]
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        m = self.act(self.c_fc(x))
        m = self.c_proj(m)

        if self.mp_comm is not None:
            m = self.mp_comm.all_reduce(m, "mlp")
        return self.resid_dropout(m)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_ctx,
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn="quick_gelu",
        zero_out=False,
        init_scale=1.0,
        res_scale=1.0,
        m_attn=0.25,
        m_mlp=1.0,
        mp_comm: Comm = None,
        key_bias=False,
    ):
        super().__init__()

        self.attn: Attention = Attention(
            d_model=d_model,
            n_ctx=n_ctx,
            d_attn=int(m_attn * d_model),
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            zero_out=zero_out,
            init_scale=init_scale,
            mp_comm=mp_comm,
            key_bias=key_bias,
        )

        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(
            d_model=d_model,
            d_ff=int(m_mlp * d_model),
            resid_dropout=resid_dropout,
            afn=afn,
            zero_out=zero_out,
            init_scale=init_scale,
            mp_comm=mp_comm,
        )
        self.ln_2 = LayerNorm(d_model)
        self.res_scale = res_scale

    def forward(self, inputs, hidden_state=None):
        """
        output is x + a + m, where a = attn(ln(x)), m = mlp(ln(x+a))

        :param x: activations
        :param hidden_state: past activations for fast sampling with cached activations
        """

        inputs_normalized = self.ln_1(inputs)

        attention_result, output_hidden_state = self.attn(
            inputs_normalized, hidden_state=hidden_state
        )

        attention_residual = inputs + attention_result

        attention_residual_normalized = self.ln_2(attention_residual)

        mlp_output = self.mlp(attention_residual_normalized)

        # we use unnormalized inputs to all functions for residuals
        final_residual = inputs + self.res_scale * (attention_result + mlp_output)
        return final_residual, output_hidden_state


Self = TypeVar("Self")


class HiddenState(ABC):
    @abstractmethod
    def detach(self: Self) -> Self:
        ...

    @abstractmethod
    def concat_with(self: Self, new_hidden_state: Self) -> Self:
        """Concatenates the two hidden states along the time dimension"""
        ...


@dataclass
class PastKVHiddenState(HiddenState):
    hidden_ctx_len: int
    resblock_hidden_states: List[torch.Tensor]

    def detach(self) -> "PastKVHiddenState":
        return PastKVHiddenState(
            hidden_ctx_len=self.hidden_ctx_len,
            resblock_hidden_states=[
                hidden_state.detach() if hidden_state is not None else None
                for hidden_state in self.resblock_hidden_states
            ],
        )

    def _cat_ignore_none(self, tensors, *args, **kwargs):
        """torch cats, but excludes None arguments"""
        return torch.cat([tensor for tensor in tensors if tensor is not None], *args, **kwargs)

    def concat_with(self, new_hidden_state: "PastKVHiddenState") -> "PastKVHiddenState":
        """
        Takes the tensors in the current hidden state and the new hidden state
        and concatenates them along the sequence dimension.
        """
        past_hidden_state = self
        return PastKVHiddenState(
            hidden_ctx_len=past_hidden_state.hidden_ctx_len + new_hidden_state.hidden_ctx_len,
            resblock_hidden_states=[
                self._cat_ignore_none((past, new), dim=-2)
                for past, new in zip(
                    past_hidden_state.resblock_hidden_states,
                    new_hidden_state.resblock_hidden_states,
                )
            ],
        )


class TransformerTorso(nn.Module):
    def __init__(
        self,
        d_model,
        n_ctx,
        n_head,
        n_depth,  # The number of resblocks.
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn="quick_gelu",
        zero_out=False,
        init_scale=1.0,
        res_scale=False,
        m_attn=0.25,
        m_mlp=1.0,
        mp_comm: Comm = None,
        key_bias=False,
    ):
        super().__init__()
        self.n_ctx = n_ctx

        res_scale = 1.0 / n_depth if res_scale else 1.0

        self.resblocks = nn.ModuleList()
        for resblock_idx in range(n_depth):
            resblock = ResidualAttentionBlock(
                d_model=d_model,
                n_ctx=n_ctx,
                n_head=n_head,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                afn=afn,
                zero_out=zero_out,
                init_scale=init_scale,
                res_scale=res_scale,
                m_attn=m_attn,
                m_mlp=m_mlp,
                mp_comm=mp_comm,
                key_bias=key_bias,
            )
            self.resblocks.append(resblock)

    def forward(self, x, hidden_states=None):
        """
        x:              input tensor
        hidden_state:   list of hidden_states, one for each resblock
                        if None, then becomes [None] * n_layer

        returns
        x:                      The output of the layers
        output_hidden_states:   A list of size self.resblocks with each hidden_state
        """
        if hidden_states is None:
            hidden_states = [None] * len(self.resblocks)
        else:
            hidden_states = hidden_states

        assert len(hidden_states) == len(
            self.resblocks
        ), f"number of hidden states should match number of resblocks: {len(hidden_states)} != {len(self.resblocks)}"

        output_hidden_states = []
        # Blocks
        for l, hidden_state in zip(self.resblocks, hidden_states):
            x, output_hidden_state = l(x, hidden_state=hidden_state)
            output_hidden_states.append(output_hidden_state)

        return x, output_hidden_states


def get_normal(*shape, std=0.01):
    w = torch.empty(shape)
    nn.init.normal_(w, std=std)
    return w


class PositionEmbedding(nn.Module):
    def __init__(self, n_ctx, d_model, init_scale=1.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.weight = nn.Parameter(get_normal(n_ctx, d_model, std=0.01 * init_scale))

    def forward(self, start_pos, end_pos):
        assert end_pos <= self.n_ctx, (
            f"Positional embedding doesnt exist for position {end_pos}. This is likely due to "
            f"feeding a sequence that's longer than the context length n_ctx={self.n_ctx}."
        )
        return self.weight[start_pos:end_pos]


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx,
        n_vocab,
        d_model=128,
        n_layer=2,
        heads=1,
        attn_dropout=0.0,
        resid_dropout=0.0,
        emb_dropout=0.0,
        zero_out=False,
        init_scale=1.0,
        res_scale=False,
        m_attn=0.25,
        m_mlp=1,
        mp_comm: Comm = None,
        include_pos_embeddings=True,
        include_input_embeddings=True,
        include_output_unembeddings=True,
        # For e.g. reward model training, we want the final layer norm before the extra head
        # but not the output unembeddings, so control this separately.
        include_final_layer_norm=True,
        afn="quick_gelu",
        key_bias=False,
        flatten_multi_index_batch_dims=False,  # allows the first dims to be batch dims and flattens them, outputs will be flattened
        global_idxs_for_resblocks: Optional[List[int]] = None,
        **extra_args,
    ):
        """
        Autoregressive Transformer with various bells and whistles. We follow the same
        model-parallel sharding scheme as the one described in https://nv-adlr.github.io/MegatronLM
        """
        super().__init__()
        self.n_ctx = n_ctx
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.depth = n_layer
        self.include_input_embeddings = include_input_embeddings
        self.include_pos_embeddings = include_pos_embeddings
        self.include_final_layer_norm = include_final_layer_norm
        self.include_output_unembeddings = include_output_unembeddings
        self.mp_comm = mp_comm  # ShardedModel must expose an mp_comm object
        self._global_idxs_for_resblocks = global_idxs_for_resblocks

        self.flatten_multi_index_batch_dims = flatten_multi_index_batch_dims

        self.is_using_dropout = bool(attn_dropout or resid_dropout or emb_dropout)

        n_shards = 1 if mp_comm is None else mp_comm.size
        assert d_model % n_shards == 0

        # Round to 128 for tensorcores
        self.n_vocab_padded = int(np.ceil(n_vocab / 128)) * 128

        if include_input_embeddings:
            # No padding needed here, since no matmuls are involved
            # Keeping for backwards-compatibility, but could change this if desired
            input_n_vocab = self.n_vocab_padded

            self.embedding = nn.Embedding(input_n_vocab, exact_div(d_model, n_shards))

            nn.init.normal_(self.embedding.weight, std=0.02 * init_scale)
            self.embed_dropout = nn.Dropout(emb_dropout)

            if self.include_pos_embeddings:
                self.position_embedding = PositionEmbedding(
                    n_ctx=n_ctx, d_model=exact_div(d_model, n_shards), init_scale=init_scale
                )
                self.pos_emb_dropout = nn.Dropout(emb_dropout)

        self.torso = TransformerTorso(
            d_model=d_model,
            n_ctx=n_ctx,
            n_head=heads,
            n_depth=n_layer,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            afn=afn,
            zero_out=zero_out,
            init_scale=init_scale,
            res_scale=res_scale,
            m_attn=m_attn,
            m_mlp=m_mlp,
            mp_comm=mp_comm,
            key_bias=key_bias,
        )

        if include_final_layer_norm:
            self.ln_f = LayerNorm(d_model)

        if include_output_unembeddings:
            # Note: We use n_vocab_padded here so that we can use TensorCores for the big matmul and softmax
            x_out_weight = nn.Embedding(self.n_vocab_padded, exact_div(d_model, n_shards)).weight
            nn.init.normal_(x_out_weight, std=0.02 * init_scale)
            self.unembedding_weights = x_out_weight

    def forward(
        self, x, hidden_state: Optional[PastKVHiddenState] = None, act_dtype=torch.float16
    ) -> Dict[str, torch.Tensor]:
        """
        :param x: activations or tokens
        :param hidden_state: hidden state from a model invocation earlier in the sequence
        :param act_dtype: the dtype of the activations (either fp16 or fp32)
        :return: A dictionary with either "acts" or "logits"
        """

        if hidden_state is None:
            hidden_ctx_len = 0
            resblock_hidden_states = None
        else:
            hidden_ctx_len = hidden_state.hidden_ctx_len
            resblock_hidden_states = hidden_state.resblock_hidden_states

        assert x.numel() > 0, "Need at least one sequence element"

        if self.include_input_embeddings:
            assert x.dim() >= 1, f"need atleast n_ctx dims: {x.shape}"
            if self.flatten_multi_index_batch_dims:
                # batch so we have two dimensions
                n_ctx = x.shape[-1]
                x = x.reshape(-1, n_ctx)
            else:
                assert x.dim() == 2, f"expecting (batch, n_ctx): {x.shape}"

            x = self.embed_tokens(x, ctx_len=hidden_ctx_len, act_dtype=act_dtype)

        assert x.dim() == 3, f"expect activations of size [n_batch, n_ctx, n_dim]: {x.shape}"

        # Transformer
        x, output_hidden_states = self.torso(x, hidden_states=resblock_hidden_states)
        assert x.dtype == act_dtype, x.dtype

        # we expect the output_hidden_state to be a list of length self.depth
        # with dtype act_dtype
        assert isinstance(
            output_hidden_states, list
        ), f"Expected list output for hidden state, got {type(output_hidden_states)}"
        assert len(output_hidden_states) == self.depth, "expected hidden state for each layer"
        assert all(
            state.dtype == act_dtype for state in output_hidden_states if state is not None
        ), f"got wrong dtype for hidden_state: {output_hidden_states[0].dtype}"

        if self.include_final_layer_norm:

            x = self.ln_f(x)

            # Make sure the output is the same at each shard, since the layer norm params can drift
            if self.mp_comm is not None:
                x = self.mp_comm.broadcast(x, src=self.mp_comm.ranks[0], name="final")

        outputs = dict(
            acts=x,
            hidden_state=PastKVHiddenState(
                hidden_ctx_len=x.size(1), resblock_hidden_states=output_hidden_states
            ),
        )
        if self.include_output_unembeddings:
            if self.mp_comm is not None:
                x = torch.split(x, x.size(-1) // self.mp_comm.size, dim=(-1))[self.mp_comm.my_index]

            logits = F.linear(x.type(act_dtype), self.unembedding_weights.type(act_dtype))

            # Mask out padding logits
            logits[:, :, self.n_vocab :] = -torch.finfo(logits.dtype).max

            if self.mp_comm is not None:
                logits = self.mp_comm.all_reduce(logits, "logits")

            outputs["logits"] = logits

        return outputs

    def embed_tokens(self, x, ctx_len=0, act_dtype=torch.float16):
        tokens = x
        assert isinstance(
            tokens, (torch.LongTensor, torch.cuda.LongTensor)
        ), f"Tokens should be type long: {getattr(tokens, 'dtype', type(tokens))}"
        assert (0 <= tokens).all() and (
            tokens < self.n_vocab
        ).all(), f"{tokens.max()} >= {self.n_vocab} or {tokens.min()} < 0"

        emb = F.embedding(
            tokens,
            self.embedding.weight,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        ).type(act_dtype)

        if self.include_pos_embeddings:
            pos_emb = self.position_embedding(
                torch.tensor(ctx_len), torch.tensor(ctx_len + tokens.size(-1))
            ).type(act_dtype)

            embedded_tokens = self.embed_dropout(emb) + self.pos_emb_dropout(pos_emb)
        else:
            embedded_tokens = self.embed_dropout(emb)

        if self.mp_comm is not None:
            comm = self.mp_comm
            if len(comm.ranks) == 1:
                result = [embedded_tokens]
            else:
                result = comm.all_gather_no_backward(embedded_tokens, "input_gather")
            embedded_tokens = torch.cat(result, dim=-1)
        return embedded_tokens

    def act_shape(self, in_shape):
        """Gives the shape of intermediate activations for this model.
        Notably, this might not match the output shape (if
        self.include_output_unembeddings).

        Important: in_shape is the input shape of the **tokens** to the model,
        not to this module.
        """
        act_shape = tuple(in_shape)
        act_shape += (self.d_model,)

        if self.flatten_multi_index_batch_dims:
            # flatten the batch dims
            act_shape = (np.prod(act_shape[:-2]), *act_shape[-2:])
        assert (
            len(act_shape) == 3
        ), f"Expecting all intermediate act shapes to be 3 dims but got {len(act_shape)}"
        return act_shape


def print_hparams(hparams, title="Hparams"):
    print(f"{title}:")
    for key, value in sorted(hparams.items()):
        print(f"{key}: {value}")


def build_with_random_weights(
    layout: ModelLayout, n_vocab: int, device, model_H: Hyperparams = None, verbose=None
) -> Transformer:
    """
    Instantiate a new Transformer model piece with our specifications

    :param layout:
    :param n_vocab:
    :param device:
    :param model_H: This should include both the static and dynamic model_H
    :return: a Transformer model_piece that corresponds to this rank
    """
    is_logging_rank = layout.is_logging_rank

    if is_logging_rank:
        print("Constructing Transformer with the following model hparams:")
        print_hparams(model_H, "model_H")

    if model_H.d_model % 512 != 0 and torch.device(device).type == "cuda":
        logging.warning(
            f"d_model of {model_H.d_model} is not divisible by 512 (8*8*8), this means that "
            f"tensorcores won't be used in attention (CUDA cores will be used instead) and this "
            f"will results in a ~20% slowdown"
        )

    mp_comm = create_model_parallel_comm(layout)

    # Only pass these params to the Transformer constructor if they're in model_H. That way, the
    # defaults specified in the Transformer constructor apply.
    optional_hparam_keys = [
        "zero_out",
        "init_scale",
        "res_scale",
        "afn",
        "include_pos_embeddings",
        "resid_dropout",
        "attn_dropout",
        "emb_dropout",
        "key_bias",
    ]
    optional_hparams = {k: v for (k, v) in model_H.items() if k in optional_hparam_keys}

    model = Transformer(
        n_ctx=model_H.get("max_n_ctx") or model_H.n_ctx,
        n_vocab=n_vocab,
        d_model=model_H.d_model,
        n_layer=model_H.n_layer,
        heads=model_H.heads,
        mp_comm=mp_comm,
        m_attn=model_H.m_attn,
        m_mlp=model_H.m_mlp,
        verbose=verbose and is_logging_rank,
        **optional_hparams,
    )

    # Convert before moving to CUDA -- don't want to OOM
    maybe_convert_weights(
        model,
        fp16_conv_weights=model_H.fp16_conv_weights,
        fp16_embedding_weights=model_H.fp16_embedding_weights,
    )

    model = model.to(device)

    return model


def maybe_convert_weights(model, *, fp16_conv_weights: bool, fp16_embedding_weights: bool):
    """
    Scans through the parameters in the model and converts Conv1D and/or Embedding parameters to
    float16 depending on fp16_conv_weights and fp16_embedding_weights.
    """
    if fp16_conv_weights:

        def _convert_conv_weights_to_fp16(l):
            if isinstance(l, Conv1D):
                l.weight.data = l.weight.data.half()

        model.apply(_convert_conv_weights_to_fp16)

    if fp16_embedding_weights:

        def _convert_embedding_weights_to_fp16(l):
            if isinstance(l, torch.nn.Embedding):
                l.weight.data = l.weight.data.half()

        model.apply(_convert_embedding_weights_to_fp16)
