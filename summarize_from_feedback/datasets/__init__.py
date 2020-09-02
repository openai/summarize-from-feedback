import random

import torch

from summarize_from_feedback.datasets.cnndm import (
    cnndm_generator,
    cnndm_filtered_generator,
    cnndm_filtered_generator_short,
)
from summarize_from_feedback.datasets.test import test_generator
from summarize_from_feedback.datasets.tldr import (
    tldr_filtered_generator,
    tldr_filtered_queries_generator,
)
from summarize_from_feedback.utils import even_more_itertools

_DATASETS = {
    "tldr_3_filtered": tldr_filtered_generator,
    "tldr_3_filtered_queries": tldr_filtered_queries_generator,
    "test": test_generator,
    "cnndm": cnndm_generator,
    "cnndm_filtered": cnndm_filtered_generator,
    "cnndm_filtered_short": cnndm_filtered_generator_short,
}


def get_dataset(name, split, layout, repeat=True, seed=None):
    if seed is None:
        seed = torch.initial_seed()

    data = list(_DATASETS[name](split))

    def shuffled():
        my_random = random.Random(seed)
        while True:
            my_random.shuffle(data)
            yield from data
            if not repeat:
                return

    return even_more_itertools.distribute(shuffled(), layout=layout)
