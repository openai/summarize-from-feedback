import functools

import torch

from summarize_from_feedback import tasks
from summarize_from_feedback.query_response_model import QueryResponseModel, PADDING_TOKEN
from summarize_from_feedback.utils.torch_utils import first_true_indices, gather_one
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq


def _response_indices(response_tokens):
    indices = first_true_indices(response_tokens == PADDING_TOKEN) - 1
    return torch.max(indices, torch.zeros([1], dtype=indices.dtype, device=response_tokens.device))


def _wrap_reward_model_fn(fn):
    @functools.wraps(fn)
    def wrapped(outputs_mb, inputs_mb):
        rewards = outputs_mb["reward"]["response"][:, :, 1:]
        rewards = gather_one(rewards, inputs_mb["last_response_index"], dim=2)
        outputs_mb["reward"] = rewards
        return fn(outputs_mb, inputs_mb)

    return wrapped


class RewardModel(QueryResponseModel):
    """
    Represents a reward model, containing a reward head.
    Only a single reward is computed for each sequence.
    """

    def __init__(self, task_hparams: tasks.TaskHParams = None, init_zero=False, **kwargs):
        init_scales = kwargs.pop("init_scales", dict())
        if init_zero:
            assert "reward" not in init_scales
            init_scales["reward"] = 0
        super().__init__(logit_head=False, heads=("reward",), init_scales=init_scales, **kwargs)
        self.task_hparams = task_hparams

    def reward(self, query_tokens, response_tokens, eval_fn=None, eval_inputs=None, **kwargs):
        """
        :return: A dict with structure:
            reward: [batch, num_responses]
            eval_stats: dict of stats returned by eval_fn
        """
        last_response_indices = _response_indices(response_tokens).to(self.device)
        if self.task_hparams is not None:
            assert_eq(query_tokens.size(1), self.task_hparams.query.length)
            assert_eq(response_tokens.size(2), self.task_hparams.response.length)
        assert query_tokens.size(0) == response_tokens.size(0)
        if eval_fn is not None:
            eval_fn = _wrap_reward_model_fn(eval_fn)
            eval_inputs["last_response_index"] = last_response_indices
        result = self._eval(
            query_tokens, response_tokens, eval_fn=eval_fn, eval_inputs=eval_inputs, **kwargs
        )
        result["reward"] = gather_one(
            result["reward"]["response"][:, :, 1:], last_response_indices, dim=2
        )

        assert_shape_eq(result["reward"], (response_tokens.size(0), response_tokens.size(1)))
        return result
