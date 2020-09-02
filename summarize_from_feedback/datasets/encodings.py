import json
import os.path
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import blobfile as bf
import regex as re
import torch

ENCODINGS_BASE = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/encodings"


def read_file(path):
    with bf.BlobFile(path, "rb") as f:
        return f.read()


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@dataclass
class Encoding:
    name: str
    n_vocab: int
    # End of text token
    eot_token: int = None
    registry: ClassVar[dict] = dict()
    eoprefix_token: int = None

    def __post_init__(self):
        self._register()

    def _register(self):
        assert self.name not in self.registry
        self.registry[self.name] = self

    def __call__(self, text):
        return self.encode(text)

    def encode(self, text):
        """Convert text (or other data) into an array of integer tokens"""
        raise NotImplementedError

    def decode(self, tokens) -> str:
        """Convert array of integer tokens into text (or other data)."""
        raise NotImplementedError


@dataclass
class BPEEncoding(Encoding):
    base_path: str = None
    encoder_path: str = "encoder.json"
    bpe_path: str = "vocab.bpe"
    n_denoise_sentinels: int = 0

    # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
    pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __post_init__(self):
        super().__post_init__()

        self.base_path = self.base_path or os.path.join(ENCODINGS_BASE, self.name)
        self.full_encoder_path = os.path.join(self.base_path, self.encoder_path)
        self.full_bpe_path = os.path.join(self.base_path, self.bpe_path)

        # We don't load the full BPE data until needed
        # Initialize them to 'None' here to make the linter happy
        # Note: Do not move these to @dataclass fields. That will lead to enormous outputs,
        #       since the default dataclass __repr__ prints the values of all fields.
        self._token_str_to_idx = None
        self._token_idx_to_str = None
        self.byte_decoder = None
        self.byte_encoder = None
        self.bpe_ranks = None

        # Without this cache, performance is 5x slower
        self.bpe = lru_cache(maxsize=2 ** 17)(self.bpe)

    def _load(self):
        if self._token_str_to_idx is not None:
            return

        self._token_str_to_idx = json.loads(read_file(self.full_encoder_path).decode())
        bpe_data = read_file(self.full_bpe_path).decode()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

        assert self.eot_token == self._token_str_to_idx["<|endoftext|>"]

        # Add an <|end_of_prefix|> token
        if self.eoprefix_token is not None:
            assert not self._token_str_to_idx.get("<|endofprefix|>")
            self._token_str_to_idx["<|endofprefix|>"] = self.eoprefix_token

        # Add denoise sentinel tokens like <|dn_1|> <|dn_2|> etc.
        # These tokens are added to the end of the vocabulary range
        for denoise_sentinel_idx in range(self.n_denoise_sentinels):
            str_repr = f"<|dn_{denoise_sentinel_idx}|>"
            assert not self._token_str_to_idx.get(str_repr)

            n_non_sentinel_tokens = self.n_vocab - self.n_denoise_sentinels
            sentinel_token = n_non_sentinel_tokens + denoise_sentinel_idx

            self._token_str_to_idx[str_repr] = sentinel_token

        assert len(self._token_str_to_idx) == self.n_vocab

        self._token_idx_to_str = {v: k for k, v in self._token_str_to_idx.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

    def bpe(self, token):  # pylint: disable=method-hidden
        word = tuple(self.byte_encoder[b] for b in token.encode("utf-8"))

        if len(word) == 1:
            return word

        while True:
            min_pair = None
            min_idxs = []
            min_rank = None
            for i, pair in enumerate(zip(word[:-1], word[1:])):
                try:
                    rank = self.bpe_ranks[pair]
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                        del min_idxs[::]
                        min_idxs.append(i)
                    elif min_rank == rank and i > min_idxs[-1] + 1:
                        min_idxs.append(i)
                except KeyError:
                    pass

            if min_pair is None:
                break

            new_word = []
            i_start = 0
            for i in min_idxs:
                new_word.extend(word[i_start:i])
                new_word.append(min_pair[0] + min_pair[1])
                i_start = i + 2

            if i_start < len(word):
                new_word.extend(word[i_start:])

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break

        return word

    def encode(self, text):
        self._load()
        bpe_tokens = [
            self._token_str_to_idx[bpe_token]
            for token in re.findall(self.pat, text)
            for bpe_token in self.bpe(token)
        ]
        return bpe_tokens

    def decode(self, tokens, errors="replace"):
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 1 and tokens.size(0) == 1:
                tokens = tokens.tolist()
            else:
                tokens = tokens.squeeze().tolist()

        decoded_bytes = self.decode_bytes(tokens)
        text = decoded_bytes.decode("utf-8", errors=errors)
        return text

    def decode_bytes(self, tokens):
        self._load()
        text = "".join([self._token_idx_to_str[token] for token in tokens])
        return bytearray([self.byte_decoder[c] for c in text])


Reversible = BPEEncoding(name="reversible_50000", n_vocab=50257, eot_token=50256)
