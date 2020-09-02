from summarize_from_feedback.datasets import get_dataset


def _get_first_n(seed, n, shards):
    it = iter(get_dataset("test", split="test", seed=seed))
    result = set()
    for shard in range(shards):
        result = result.union(set(next(it)["query"] for _ in range(n)))
    return result


def test_policy_basic():
    for seed in range(10):
        assert _get_first_n(seed, 4, 1) == _get_first_n(seed, 2, 2)

    assert _get_first_n(0, 4, 1) != _get_first_n(1, 4, 1)
