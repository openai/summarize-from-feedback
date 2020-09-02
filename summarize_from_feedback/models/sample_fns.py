from dataclasses import dataclass
from typing import Callable

import torch

Logits = torch.FloatTensor


@dataclass
class Sample:
    logits: Logits
    tokens: torch.LongTensor


Sampler = Callable[[Logits], Sample]


def standard(temperature: float = 1.0) -> Sampler:
    def sample(logits: Logits) -> Sample:
        logits = logits / (temperature + 1e-7)

        # There was a regression in torch that made categorical only work with fp32.
        # We can track the issue on github and remove this once it makes it into a
        # pytorch release or nightly:
        #
        # https://github.com/pytorch/pytorch/issues/29211
        #
        logits_fp32 = logits.float()

        return Sample(
            logits=logits, tokens=torch.distributions.Categorical(logits=logits_fp32).sample()
        )

    return sample


def argmax() -> Sampler:
    def sample(logits: Logits) -> Sample:
        return Sample(logits=logits, tokens=torch.argmax(logits, dim=-1))

    return sample


def nucleus_sampler(top_p: float = 0.9, temperature=1.0) -> Sampler:
    """
    Return a sampler that decides diversity via nucleus sampling.

    p=0.9 means that the top 90% of likelihood-weighted options are considered. p=0.0 is
    equivalent to argmax, p=1.0 has no effect.

    When a logit is on the boundary of being included or not being included, default
    to including it.
    """
    if top_p == 0.0:
        return argmax()

    if top_p == 1.0:
        return standard(temperature=temperature)

    def sample(logits: Logits) -> Sample:
        """
        Remove logits that do not represent the top_p proportion of likelihoods.

        When a logit is on the boundary of being included or not being included, default
        to including it.
        """
        logits = logits.clone()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")
        return standard(temperature=temperature)(logits)

    return sample
