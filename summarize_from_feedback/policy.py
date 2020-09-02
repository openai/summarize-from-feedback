import functools

from summarize_from_feedback import tasks
from summarize_from_feedback.query_response_model import QueryResponseModel


def _wrap_policy_fn(fn, heads=()):
    @functools.wraps(fn)
    def wrapped(outputs_mb, inputs_mb):
        for key in heads:
            outputs_mb[key] = outputs_mb[key]["response"]
        return fn(outputs_mb, inputs_mb)

    return wrapped


class Policy(QueryResponseModel):
    """
    Represents a RL policy + value function, containing a value and logit head.
    Only returns pre-token values (never evaluates on the final response token)
    """

    def __init__(self, task_hparams: tasks.TaskHParams = None, logit_head=True, **kwargs):
        super().__init__(logit_head=logit_head, heads=("value",), **kwargs)
        self.task_hparams = task_hparams

    def sample(
        self, query_tokens, partial_responses=None, responses_per_query=1, sample_len=None, **kwargs
    ):
        """
        Samples from the policy given the context (query_tokens). partial_responses, if provided,
        should be a torch tensor of size (batch_size, responses_per_query, X) where X = response length so far.
        :return: A dict with structure:
            tokens: [batch, num_responses, sample_len]
            logprobs: [batch, num_responses, sample_len]
            value: [batch, num_responses, sample_len]
        """

        response_len = 0  # length of responses so far
        if partial_responses is not None:
            response_len = partial_responses.size(2)

        assert self.logit_head

        # sample_len is length of sample to be completed
        if sample_len is None:
            assert self.task_hparams is not None
            sample_len = self.task_hparams.response.length - response_len

        if self.task_hparams is not None:
            assert query_tokens.size(1) == self.task_hparams.query.length, f"{query_tokens.size()}"

        ret = self._sample(
            query_tokens,
            partial_responses=partial_responses,
            sample_len=sample_len,
            responses_per_query=responses_per_query,
            **kwargs,
        )
        for key in self.heads:
            ret[key] = ret[key]["response"]

        return ret

    def eval(self, query_tokens, response_tokens, eval_fn=None, **kwargs):
        """
        :return: A dict with structure:
            value: [batch, num_responses, response_length]
            eval_stats: dict of stats returned by eval_fn
        """
        if self.task_hparams is not None:
            assert query_tokens.size(1) == self.task_hparams.query.length, f"{query_tokens.size()}"
            assert (
                response_tokens.size(2) == self.task_hparams.response.length
            ), f"{response_tokens.size()}"
        assert query_tokens.size(0) == response_tokens.size(0)
        if eval_fn is not None:
            eval_fn = _wrap_policy_fn(
                eval_fn, list(self.heads) + (["logits"] if self.logit_head else [])
            )
        ret = self._eval(query_tokens, response_tokens[:, :, :-1], eval_fn=eval_fn, **kwargs)
        for key in self.heads:
            ret[key] = ret[key]["response"]
        return ret
