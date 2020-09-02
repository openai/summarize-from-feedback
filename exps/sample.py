#!/usr/bin/env python3

import fire

from summarize_from_feedback import sample
from summarize_from_feedback.utils import experiment_helpers as utils
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher


def experiment_definitions():
    sup_xl = combos(
        bind_nested("model_spec", utils.sup4()),
        bind_nested("task", utils.tldr_task),
        bind("query_dataset_split", "valid"),
        bind("mpi", 1),
        bind("num_queries", 128),
        bind("sample.temperature", 0.01),
    )

    ppo_xl = combos(
        bind_nested("model_spec", utils.sup4_ppo_rm4()),
        bind_nested("task", utils.tldr_task),
        bind("query_dataset_split", "valid"),
        bind("mpi", 1),
        bind("num_queries", 128),
        bind("sample.temperature", 0.01),
    )

    test = combos(
        bind_nested("task", utils.test_task),
        bind_nested("model_spec", utils.random_teeny_model_spec(n_shards=2)),
        bind_nested("orig_model_spec", utils.random_teeny_model_spec(n_shards=2)),
        bind("query_dataset_split", "train"),
        bind("mpi", 2),
        bind("num_queries", 8),
        bind("responses_per_query", 2),
        bind("responses_per_query_per_batch", 1),
    )

    test_cpu = combos(
        test,
        bind("model_spec.device", "cpu"),
        bind("orig_model_spec.device", "cpu"),
        bind("fp16_activations", False),
    )
    return locals()


if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(experiment_dict=experiment_definitions(), main_fn=sample.main)
    )

"""
python summarize_from_feedback/exps/sample.py test test 2>&1 | tee log
python summarize_from_feedback/exps/sample.py tldr_main tldr-sample-$(date +%y%m%d%H%M)
"""
