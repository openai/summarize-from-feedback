import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, List, Dict

import blobfile as bf
import numpy as np
import torch
from mpi4py import MPI

import summarize_from_feedback
from summarize_from_feedback.model_layout import ModelLayout
from summarize_from_feedback.models import sample_fns
from summarize_from_feedback.models.loss_functions import softmax_xent_loss_fn
from summarize_from_feedback.models.transformer import Hyperparams
from summarize_from_feedback.models.transformer import build_with_random_weights
from summarize_from_feedback.utils import exact_div, hyperparams
from summarize_from_feedback.utils import blobs
from summarize_from_feedback.utils.dist_utils import (
    setup_cuda_device_and_dist,
    create_data_parallel_comm,
    create_within_replica_comm,
)
from summarize_from_feedback.utils.nested import map_nested
from summarize_from_feedback.utils.torch_utils import nans, to_numpy


@dataclass
class RunParams(hyperparams.HParams):
    fp16_embedding_weights: bool = False
    fp16_conv_weights: bool = False
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    emb_dropout: float = 0.0

    n_shards: int = 1

    def all_gpu_layout(self):
        return ModelLayout.standard(
            n_shards=self.n_shards,
            total_gpus=MPI.COMM_WORLD.Get_size(),
            my_rank=MPI.COMM_WORLD.Get_rank(),
        )


def sample(
    self,
    contexts,
    sample_len,
    sample_fn,
    act_dtype=torch.float16,
    model_output_keys=(),
    **model_call_kwargs,
):
    assert not self.training
    n_batch, n_ctx = contexts.shape
    with torch.no_grad():
        tokens = []
        logprobs = []
        extra_outputs = []
        output = self(contexts, act_dtype=act_dtype, **model_call_kwargs)
        past_hidden_state = output["hidden_state"].detach()
        prev_logits = output["logits"][:, -1:, :]
        for sample_t in range(n_ctx, n_ctx + sample_len):
            new = sample_fn(prev_logits)
            new_tokens, new_logits = new.tokens, new.logits
            new_logprobs = -softmax_xent_loss_fn(
                dict(logits=new_logits.float()), dict(targets=new_tokens), reduction="none"
            )
            assert new_tokens.shape == (n_batch, 1)
            assert new_logprobs.shape == (n_batch, 1)
            tokens.append(new_tokens)
            logprobs.append(new_logprobs)
            extra_outputs.append({k: output[k] for k in model_output_keys})

            # NOTE: last iteration is thrown away
            output = self(
                new_tokens, hidden_state=past_hidden_state, act_dtype=act_dtype, **model_call_kwargs
            )
            prev_logits = output["logits"]

            past_hidden_state = past_hidden_state.concat_with(output["hidden_state"].detach())

        tokens = torch.cat(tokens, dim=1)
        logprobs = torch.cat(logprobs, dim=1)
        extra_outputs = {
            k: torch.cat([extra[k] for extra in extra_outputs], dim=1) for k in model_output_keys
        }
    return dict(tokens=tokens, logprobs=logprobs, **extra_outputs)


class ModelWithHeads(torch.nn.Module):
    def __init__(self, model, scalar_heads, d_model, init_scales=1.0):
        super().__init__()
        self.model = model
        self.scalar_head_names = scalar_heads
        if not isinstance(init_scales, dict):
            init_scales = {head_name: init_scales for head_name in scalar_heads}

        self.scalar_heads = torch.nn.ModuleDict()
        for name in self.scalar_head_names:
            head = torch.nn.Linear(d_model, 1)
            init_std = init_scales.get(name, 1.0) / np.sqrt(d_model + 1)
            torch.nn.init.normal_(head.weight, std=init_std)
            torch.nn.init.zeros_(head.bias)
            self.scalar_heads[name] = head

        for attr in [
            "include_input_embeddings",
            "embedding",
            "include_pos_embeddings",
            "position_embedding",
            "include_final_layer_norm",
            "include_output_unembeddings",
            "ln_f",
            "unembedding_weights",
            "torso",
            "mp_comm",
            "n_ctx",
        ]:
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        x = outputs["acts"]
        for name, head in self.scalar_heads.items():
            outputs[name] = torch.squeeze(head(x.type(head.weight.dtype)), dim=-1)
        return outputs

    def act_shape(self, in_shape):
        return self.model.act_shape(in_shape)


@dataclass
class ModelSpec(hyperparams.HParams):
    device: str = "cuda"
    load_path: str = None
    use_cache: bool = True
    short_name: Optional[str] = None
    init_heads: Optional[List[str]] = None
    map_heads: Dict[str, str] = field(default_factory=dict)
    run_params: RunParams = field(default_factory=RunParams)

    def name(self):
        if self.short_name is not None:
            return self.short_name
        elif self.load_path is not None:
            return self.load_path
        else:
            raise NotImplementedError


def save_exported_model(layout, model, model_H: Hyperparams, save_dir, save_heads: Set[str]):
    """
    Exporting a model allows it to be run with a different layout than it was trained with.

    Currently, uploading/loading an exported model is slower than saving/restoring a checkpoint,
    but if we can get exporting to be sufficiently fast, then we could replace legacy_checkpoints.py with
    this "exporting" approach.
    """
    if blobs.is_blob_url(save_dir):
        local_dir = os.path.join("/tmp", str(uuid.uuid4()))
    else:
        local_dir = save_dir
    os.makedirs(os.path.join(local_dir, "checkpoint"), exist_ok=True)

    def export_fine_piece(fine_model_piece_dict: dict, chkpt_prefix: str):
        fine_piece_path = os.path.join(
            local_dir, "checkpoint", f"{chkpt_prefix}_shard_{layout.shard_idx:03d}.pkl"
        )
        # print(f"Uploading fine_piece: {fine_piece_path}")
        torch.save(fine_model_piece_dict, fine_piece_path)
        torch.cuda.synchronize()  # Verify that the piece has finished being written

    # Export the embeddings
    if model.include_input_embeddings:
        export_fine_piece(model.embedding.state_dict(), "input_embeddings")

        if model.include_pos_embeddings:
            export_fine_piece(model.position_embedding.state_dict(), "position_embedding")
    # Export the resblocks
    for resblock_idx, resblock in enumerate(model.torso.resblocks):
        export_fine_piece(resblock.state_dict(), f"resblock_{resblock_idx:04d}")
    # Export the final_layer_norm
    if model.include_final_layer_norm:
        export_fine_piece(model.ln_f.state_dict(), "final_layer_norm")
    # Export the unembeddings
    if model.include_output_unembeddings:
        export_fine_piece({"unembedding_weights": model.unembedding_weights}, "output_unembeddings")
    for head in save_heads:
        export_fine_piece(model.scalar_heads[head].state_dict(), f"output_head_{head}")
    if blobs.is_blob_url(save_dir):
        blobs.parallel_copy_recursive(local_dir, save_dir)
        shutil.rmtree(local_dir)


def _matches_any_prefix(x, prefixes):
    return any([x.startswith(prefix) for prefix in prefixes])


def dim_to_shard(name: str) -> Optional[int]:
    if name.startswith("scalar_heads."):
        # heads should be the same on all shards
        return None
    return parameter_name_to_sharding_dim(name)


def parameter_name_to_sharding_dim(name: str) -> Optional[int]:
    """
    :returns: None if all parameters are same on all shards, otherwise the dimension to
        split upon.
    """
    if name in ["embedding.weight", "position_embedding.weight", "unembedding_weights"]:
        return -1
    if name.startswith("torso.resblocks"):
        match = re.search(r"torso\.resblocks\.\d+\.(.*)", name)
        torso_part = match.group(1)
        if torso_part.startswith("ln_1.") or torso_part.startswith("ln_2."):
            return None
        if _matches_any_prefix(
            torso_part, ["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc"]
        ):
            return -1
        if _matches_any_prefix(torso_part, ["attn.c_proj.weight", "mlp.c_proj.weight"]):
            return -2
        if _matches_any_prefix(torso_part, ["attn.c_proj.bias", "mlp.c_proj.bias"]):
            return None
        raise RuntimeError(f"Unexpected parameter name: {name}")
    if name in ["ln_f.weight", "ln_f.bias"]:
        return None
    raise RuntimeError(f"Unexpected parameter name: {name}")


def get_shard_fix_factor(name: str, model_H: Hyperparams, old_model_H: Hyperparams) -> float:
    # Hack to fix some bugs with our sharding code
    if name.startswith("torso.resblocks"):
        match = re.search(r"torso\.resblocks\.\d+\.(.*)", name)
        torso_part = match.group(1)

        # bias is added before all-reduce, which means with more shards, the
        # weights are closer to 0 than expected
        if _matches_any_prefix(torso_part, ["attn.c_proj.bias", "mlp.c_proj.bias"]):
            return float(old_model_H.n_shards) / model_H.n_shards

        if (
            _matches_any_prefix(
                torso_part,
                [
                    "attn.q_proj.weight",
                    "attn.k_proj.weight",
                    "attn.q_proj.bias",
                    "attn.k_proj.bias",
                ],
            )
            and old_model_H.use_blocksparse_attn
        ):
            return np.sqrt(np.sqrt(float(old_model_H.n_shards) / old_model_H.heads))
    return 1.0


def load_exported_model(
    layout: ModelLayout,
    model,
    model_H: Hyperparams,
    load_path: str,
    load_heads_map: Dict[str, str],
    use_cache: bool = False,
):
    """
    :param load_heads_map: maps name in model -> name to load from
    """
    if use_cache and blobs.is_blob_url(load_path):
        load_path = blobs.download_directory_cached(load_path)

    with bf.BlobFile(os.path.join(load_path, "info.json")) as f:
        info = json.load(f)

    old_model_H = Hyperparams(**info["model_hparams"])
    original_n_shards = old_model_H.n_shards
    if "n_shards" in info:
        assert info["n_shards"] == original_n_shards
    assert layout.n_shards == model_H.n_shards
    # print("orig n_shards", original_n_shards, "new n_shards", layout.n_shards)

    def fetch_single_piece(fine_piece_name):
        with bf.BlobFile(os.path.join(load_path, "checkpoint", fine_piece_name), "rb") as f:
            return torch.load(f, map_location=torch.device("cpu"))

    if original_n_shards % layout.n_shards == 0:
        n_chkpt_shards_per_rank = exact_div(original_n_shards, layout.n_shards)
        shard_idx_start = n_chkpt_shards_per_rank * layout.shard_idx
        load_shard_idxs = range(shard_idx_start, shard_idx_start + n_chkpt_shards_per_rank)

        def fetch(chkpt_prefix: str, module_name: str = ""):
            sharded_pieces = [
                fetch_single_piece(f"{chkpt_prefix}_shard_{shard_idx:03d}.pkl")
                for shard_idx in load_shard_idxs
            ]

            model_piece = {}
            for k in sharded_pieces[0].keys():
                parameter_name = ".".join([module_name, k]) if module_name else k

                sharding_dim = dim_to_shard(parameter_name)

                if sharding_dim is None:
                    val = sharded_pieces[0][k]
                else:
                    val = torch.cat([piece[k] for piece in sharded_pieces], dim=sharding_dim)
                fix_factor = get_shard_fix_factor(parameter_name, model_H, old_model_H)
                model_piece[k] = (val.float() * fix_factor).to(val.dtype)

            return model_piece

    elif layout.n_shards % original_n_shards == 0:
        n_ranks_per_chkpt_shard = exact_div(layout.n_shards, original_n_shards)
        shard_idx_to_load = layout.shard_idx // n_ranks_per_chkpt_shard
        shard_slice_idx = layout.shard_idx % n_ranks_per_chkpt_shard

        def fetch(chkpt_prefix: str, module_name: str = ""):
            unsharded_piece = fetch_single_piece(
                f"{chkpt_prefix}_shard_{shard_idx_to_load:03d}.pkl"
            )

            model_piece = {}
            for k in unsharded_piece.keys():
                parameter_name = ".".join([module_name, k]) if module_name else k

                sharding_dim = dim_to_shard(parameter_name)

                if sharding_dim is None:
                    val = unsharded_piece[k]
                else:
                    split_size = exact_div(
                        unsharded_piece[k].size()[sharding_dim], n_ranks_per_chkpt_shard
                    )
                    val = torch.split(
                        unsharded_piece[k], [split_size] * n_ranks_per_chkpt_shard, dim=sharding_dim
                    )[shard_slice_idx]
                fix_factor = get_shard_fix_factor(parameter_name, model_H, old_model_H)
                model_piece[k] = (val.float() * fix_factor).to(val.dtype)

            return model_piece

    else:
        raise NotImplementedError(
            f"Tried running a model that was originally created with "
            f"{original_n_shards} shards with {layout.n_shards} shards. The new number "
            f"of shards must evenly divide or be divisible by the original number of shards."
        )

    if model.include_input_embeddings:
        model.embedding.load_state_dict(fetch("input_embeddings", "embedding"))

        if model.include_pos_embeddings:
            model.position_embedding.load_state_dict(
                fetch("position_embedding", "position_embedding")
            )

    # fetch the resblocks
    for resblock_idx, resblock in enumerate(model.torso.resblocks):
        d = fetch(f"resblock_{resblock_idx:04d}", f"torso.resblocks.{resblock_idx}")
        if not model_H.get("key_bias"):
            d = {k: v for (k, v) in d.items() if "attn.k_proj.bias" not in k}
        resblock.load_state_dict(d)

    # fetch the final_layer_norm
    if model.include_final_layer_norm:
        model.ln_f.load_state_dict(fetch("final_layer_norm", "ln_f"))

    # fetch the unembeddings
    if model.include_output_unembeddings:
        # Pull in the one piece
        model.load_state_dict(fetch("output_unembeddings"), strict=False)

    for model_head, save_head in load_heads_map.items():
        model.scalar_heads[model_head].load_state_dict(
            fetch(f"output_head_{save_head}", f"scalar_heads.{model_head}")
        )


def _split_query_response_output_parts(x, query_length, response_padding_mask):
    """
    Given an output x with shape [batch, num_responses, query_length + response_length, *rest],
    returns a dictionary with it split into query/response parts with shapes
    [batch, query_length + 1, *rest] and [batch, num_responses, response_length + 1, *rest]
    """
    assert x.ndim >= 3
    rest_shape = x.size()[3:]
    d = dict()
    # Add this back if it's ever actually useful
    # d["query"] = torch.cat(
    #     [nans([x.size(0), 1, *rest_shape], dtype=x.dtype, device=x.device), x[:, 0, :query_length]],
    #     dim=1,
    # )
    if query_length > 0:
        d["response"] = x[:, :, query_length - 1 :]
    else:
        d["response"] = torch.cat(
            [
                nans([x.size(0), x.size(1), 1, *rest_shape], dtype=x.dtype, device=x.device),
                x[:, :, :query_length],
            ],
            dim=2,
        )
    for _ in range(len(rest_shape)):
        response_padding_mask = response_padding_mask.unsqueeze(-1)
    # fill with NaNs in places where response had padding
    d["response"].masked_fill_(
        torch.cat(
            [
                torch.zeros(
                    [x.size(0), x.size(1), 1] + [1 for _ in range(len(rest_shape))],
                    dtype=torch.bool,
                    device=x.device,
                ),
                response_padding_mask,
            ],
            dim=2,
        ),
        np.nan,
    )
    return d


PADDING_TOKEN = -1


def _zero_padding_tokens(response_tokens):
    mask = response_tokens == PADDING_TOKEN
    assert (
        not (mask[:, :, 1:] < mask[:, :, :-1]).any().item()
    ), f"Padding tokens not a suffix {to_numpy(response_tokens)}"
    return mask, torch.masked_fill(response_tokens, mask, 0)


def nested_reduce(ds, f):
    new_d = {}
    for k, v in ds[0].items():
        if isinstance(v, dict):
            new_d[k] = nested_reduce([d[k] for d in ds], f)
        else:
            new_d[k] = f([d[k] for d in ds])
    return new_d


@dataclass
class SampleHParams(hyperparams.HParams):
    temperature: float = 1.0
    top_p: float = 1.0

    def validate(self, *, prefix=""):
        assert (
            self.temperature == 1.0 or self.top_p == 1.0
        ), f"{prefix or 'SampleHParams'}: Cannot set both temperature ({self.temperature}) and top_p ({self.top_p})"

    @classmethod
    def argmax(cls):
        return cls.from_json(dict(top_p=0))


def _get_sample_fn(H: Optional[SampleHParams] = None):
    if H is None:
        H = SampleHParams()
    if H.top_p != 1.0:
        assert H.temperature == 1.0
        return sample_fns.nucleus_sampler(top_p=H.top_p)
    else:
        return sample_fns.standard(temperature=H.temperature)


class QueryResponseModel:
    """
    Handles sampling, eval, and training with shared queries.
    """

    def __init__(
        self, spec: ModelSpec, *, layout: ModelLayout, logit_head=True, heads=(), init_scales=1.0
    ):
        device = setup_cuda_device_and_dist(
            backend="nccl" if spec.device == "cuda" else "gloo",
            master_addr=None,
            device=spec.device,
        )
        self.device = device
        self.layout = layout
        assert self.layout.n_shards == spec.run_params.n_shards
        self.dp_comm = create_data_parallel_comm(layout)
        self.in_replica_comm = create_within_replica_comm(layout)

        self.logit_head = logit_head
        self.heads = heads
        self.init_scales = init_scales
        self.load(
            spec.load_path,
            run_params=spec.run_params,
            init_heads=spec.init_heads,
            map_heads=spec.map_heads,
            use_cache=spec.use_cache,
        )

        if self.device.type == "cuda":
            print(
                f"Loaded model to {self.device}. CUDA memory allocated: "
                f"{torch.cuda.memory_allocated(device=self.device) / 1e9:.2f} GB"
            )

    def _sync_params(self, params_to_init, heads_to_init):
        if self.layout.n_replicas > 1:
            for param in params_to_init:
                self.dp_comm.broadcast(
                    param.data,
                    src=self.layout.dp_sibling_ranks[0],
                    name="broadcast_params_from_zeroeth_replica",
                )

        if self.layout.n_shards > 0:
            params_to_sync_shards = []
            for head in heads_to_init:
                params_to_sync_shards.append(self.model.scalar_heads[head].weight)
                params_to_sync_shards.append(self.model.scalar_heads[head].bias)
            for param in params_to_sync_shards:
                self.model.mp_comm.broadcast(
                    param.data,
                    src=self.layout.mp_sibling_ranks[0],
                    name="broadcast_params_from_zeroeth_shard",
                )

    def _update_model_with_head_info(self, model):
        if not self.logit_head:
            model.include_output_unembeddings = False
            model.unembedding_weights = None
        model = ModelWithHeads(
            model,
            scalar_heads=list(self.heads),
            d_model=model.d_model,
            init_scales=self.init_scales,
        )
        model = model.to(self.device)
        return model

    def load(self, load_path, run_params=None, init_heads=(), map_heads={}, use_cache=False):
        """
        Rebuilds everything, but keeps API semantics: model has same layout, and is on the same device, and all heads are the same (although some may be random init)
        """
        if use_cache and blobs.is_blob_url(load_path):
            load_path = blobs.download_directory_cached(load_path)

        with bf.BlobFile(os.path.join(load_path, "info.json")) as f:
            info = json.load(f)
        self.model_hparams = Hyperparams(info["model_hparams"])
        if run_params is not None:
            extra_model_H = {k: v for k, v in run_params.to_json().items() if v is not None}
            self.model_hparams.update(**extra_model_H)
        self.encoder = summarize_from_feedback.encoder
        model = build_with_random_weights(
            layout=self.layout,
            n_vocab=self.encoder.n_vocab,
            device=self.device,
            model_H=self.model_hparams,
        )

        self.model = self._update_model_with_head_info(model)

        init_heads = set(init_heads or ())
        # Load heads from where map_heads says, or the normal head name by default
        load_heads_map = {
            head: map_heads.get(head, head) for head in self.heads if head not in init_heads
        }
        load_exported_model(
            self.layout,
            self.model,
            self.model_hparams,
            load_path,
            load_heads_map=load_heads_map,
            use_cache=use_cache,
        )
        params_to_init = []
        for head in init_heads:
            params_to_init.append(self.model.scalar_heads[head].weight)
            params_to_init.append(self.model.scalar_heads[head].bias)

        self._sync_params(params_to_init, heads_to_init=init_heads)

        self.barrier("load_finished")

    def barrier(self, name=""):
        """
        When called on all ranks, waits until all ranks are done
        """
        self.in_replica_comm.barrier(name)
        self.dp_comm.barrier(name)

    def _eval(
        self, queries, responses, eval_fn: Callable = None, eval_inputs=None, **model_call_kwargs
    ):
        """
        Run a forward pass. Return all the head values, broadcasted within each replica. If an
        eval_fn is passed, return its output across all replicas.

        :return: A dict with structure:
            eval_stats: structure from eval_fn
            [head]: {
                # disabled for now: query: [batch, query_len+1]
                response: [batch, num_responses, sample_len+1]
            }
        """
        queries = queries.to(self.device)
        responses = responses.to(self.device)
        if eval_inputs is not None:
            eval_inputs = map_nested(eval_inputs, lambda x: x.to(self.device))
        mask, responses = _zero_padding_tokens(responses)
        responses_per_query = responses.size(1)
        # NOTE: could make this more efficient by sharing context work
        tiled_queries = queries.unsqueeze(1).repeat(1, responses_per_query, 1)
        run_tokens = torch.cat([tiled_queries, responses], dim=2).flatten(0, 1)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(run_tokens, **model_call_kwargs)
        outputs_mb = dict()
        ret = dict()
        for k in list(self.heads) + (["logits"] if self.logit_head else []):
            reshaped = outputs[k].view(-1, responses_per_query, *outputs[k].size()[1:])
            d = _split_query_response_output_parts(reshaped, queries.size(1), mask)
            outputs_mb[k] = d
            if k in self.heads:
                ret[k] = d
        if eval_fn is not None:
            ret["eval_stats"] = eval_fn(outputs_mb, eval_inputs)
        return ret

    def _sample(
        self,
        context_tokens,
        sample_len,
        partial_responses=None,
        responses_per_query=1,
        sample_H=None,
        **model_call_kwargs,
    ):
        """
        :return: A dict with structure:
            samples: [batch, num_responses, sample_len]
            logprobs: [batch, num_responses, sample_len]
            [head]: {
                response: [batch, num_responses, sample_len+1]
            }
        """
        context_tokens = context_tokens.to(self.device)
        self.model.eval()
        n_batch, query_length = context_tokens.size()
        assert self.logit_head, f"Cannot sample without logit_head"
        # NOTE: could do this more efficiently by sharing context work
        repeated_context_tokens = context_tokens.unsqueeze(1).repeat(1, responses_per_query, 1)

        # Combine query and response so far into new query to be passed to _sample()
        if partial_responses is not None:
            partial_responses = partial_responses.to(self.device)
            repeated_context_tokens = torch.cat((repeated_context_tokens, partial_responses), 2)

        sample_fn = _get_sample_fn(sample_H)

        flat_context_tokens = repeated_context_tokens.flatten(0, 1)
        flat_n_batch, context_len = flat_context_tokens.shape

        assert sample_len + context_len <= self.model_hparams["n_ctx"] + 1, (
            f"Requested completion {sample_len} is too long for"
            f"context {context_len} and model context_len {self.model_hparams.n_ctx}"
        )

        results = sample(
            self.model,
            flat_context_tokens,
            sample_len=sample_len,
            sample_fn=sample_fn,
            model_output_keys=self.heads,
            **model_call_kwargs,
        )

        samples = results["tokens"]
        logprobs = results["logprobs"]
        assert samples.size(-2) == n_batch * responses_per_query
        assert logprobs.size(-2) == n_batch * responses_per_query
        assert samples.size(-1) == sample_len, f"{samples.size()} vs {sample_len}"
        assert logprobs.size(-1) == sample_len, f"{logprobs.size()} vs {sample_len}"
        samples = samples.view(n_batch, responses_per_query, sample_len)
        logprobs = logprobs.view(n_batch, responses_per_query, sample_len)

        output = dict(contexts=context_tokens, samples=samples, logprobs=logprobs)

        mask, _ = _zero_padding_tokens(output["samples"])
        # NOTE: sample doesn't return eval'ed values on final token
        mask = mask[:, :, :-1]
        for k in self.heads:
            reshaped = results[k].view(n_batch, responses_per_query, *results[k].shape[1:])
            output[k] = _split_query_response_output_parts(reshaped, query_length, mask)
        return output
