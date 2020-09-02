import pytest

from summarize_from_feedback.model_layout import ModelLayout
from summarize_from_feedback.utils.even_more_itertools import distribute


def test_shard():
    def layout(n_replicas, replica_idx):
        return ModelLayout.standard(total_gpus=n_replicas, my_rank=replica_idx)

    assert list(distribute(range(7), layout(3, 0))) == [0, 3]
    assert list(distribute(range(7), layout(3, 1))) == [1, 4]

    def dies_after_one():
        yield 5
        raise TimeoutError

    it = iter(distribute(dies_after_one(), layout(2, 0)))
    with pytest.raises(TimeoutError):
        next(it)
