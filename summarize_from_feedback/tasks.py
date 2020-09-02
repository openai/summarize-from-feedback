from dataclasses import dataclass, field
from typing import Optional, List, NewType, Union, Dict

import numpy as np
import torch

from summarize_from_feedback.utils import hyperparams
from summarize_from_feedback.utils.torch_utils import first_true_indices, to_numpy
from summarize_from_feedback.query_response_model import PADDING_TOKEN


@dataclass
class TaskQueryHParams(hyperparams.HParams):
    length: int = None
    dataset: str = None
    format_str: Optional[str] = None  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[str] = None  # defaults to repeated spaces
    pad_side: Optional[str] = None


@dataclass
class TaskResponseHParams(hyperparams.HParams):
    ref_format_str: Optional[
        str
    ] = None  # if underlying dataset yields dicts, can format arbitrarily
    length: int = None
    # Truncate response at the first occurrence of this token when sampling.
    truncate_token: Optional[int] = None


@dataclass
class TaskHParams(hyperparams.HParams):
    query: TaskQueryHParams = field(default_factory=TaskQueryHParams)
    response: TaskResponseHParams = field(default_factory=TaskResponseHParams)


# Has endoftext potentially, random stuff after
SampledTokens = NewType("SampledTokens", torch.LongTensor)
SampledTokenList = NewType("SampledTokenList", List[int])
# Has only the actual sample + padding tokens
ProcessedTokens = NewType("ProcessedTokens", torch.LongTensor)
ProcessedTokenList = NewType("ProcessedTokenList", List[int])


class ResponseEncoder:
    def __init__(self, H: TaskResponseHParams, encoder, padding_token=PADDING_TOKEN):
        self.H = H
        self.encoder = encoder
        self.padding_token = padding_token

    def process_responses(self, unprocessed_tokens: SampledTokens) -> ProcessedTokens:
        assert unprocessed_tokens.size(-1) == self.H.length
        if self.H.truncate_token is not None:
            assert self.padding_token is not None
            trunc_idxs = first_true_indices(unprocessed_tokens == self.H.truncate_token).unsqueeze(
                -1
            )
            new_size = [1] * (len(unprocessed_tokens.size()) - 1) + [self.H.length]
            idxs = torch.arange(self.H.length, device=unprocessed_tokens.device).view(*new_size)
            return torch.masked_fill(unprocessed_tokens, idxs > trunc_idxs, self.padding_token)
        else:
            return unprocessed_tokens

    def encode_response(self, text: str, allow_truncate: bool = False) -> ProcessedTokenList:
        tokens = self.encoder.encode(text)
        if allow_truncate:
            tokens = tokens[: self.H.length - (0 if self.H.truncate_token is None else 1)]
        if self.H.truncate_token is not None:
            tokens = tokens + [self.H.truncate_token]
        if self.padding_token is None:
            assert len(tokens) == self.H.length
            return tokens
        assert len(tokens) <= self.H.length, f"Response too long (limit {self.H.length}): {text}"
        return tokens + [self.padding_token] * (self.H.length - len(tokens))

    def decode_response(self, processed_response_tokens: ProcessedTokenList) -> str:
        tokens = [x for x in processed_response_tokens if x != self.padding_token]
        if self.H.truncate_token is not None:
            if tokens[-1] == self.H.truncate_token:
                tokens = tokens[:-1]
            else:
                assert len(tokens) == self.H.length
        return self.encoder.decode(tokens)

    def decode_responses(
        self, processed_response_tokens: Union[ProcessedTokens, np.ndarray]
    ):  # -> array of array of ... str:
        def _decode_responses_list(l):
            if isinstance(l[0], (int, np.int64)):
                return self.decode_response(l)
            return [_decode_responses_list(ll) for ll in l]

        return _decode_responses_list(to_numpy(processed_response_tokens))


def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    if hparams.padding is not None:
        return encoder.encode(hparams.padding)
    return encoder.encode(" ") * hparams.length


def process_query(
    query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None
):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(
            f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!"
        )
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    return dict(
        tokens=_ensure_length(
            query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence
        )
    )
