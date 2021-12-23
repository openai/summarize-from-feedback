#!/usr/bin/env python3

import fire

from summarize_from_feedback import eval_rm
from summarize_from_feedback.utils import experiment_helpers as utils
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher


def experiment_definitions():
    reward_model_spec = combos(
        bind("device", "cpu"),
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4"),
        bind("short_name", "rm4"),
    )
    rm4 = combos(
        bind_nested("task", utils.tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", reward_model_spec),
        bind("input_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/samples/sup4_ppo_rm4"),
    )

    rm4_cpu =combos(
        rm4,
        bind("fp16_activations", False),
    )
    test = combos(
        bind_nested("task", utils.test_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec()),
        bind("mpi", 1),
        bind("input_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/samples/test"),
    )
    test_cpu = combos(
        test,
        bind_nested("reward_model_spec", utils.stub_model_spec()),
    )
    tldrtest = combos(
        bind_nested("task", utils.test_tldr_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec(n_shards=2)),
        bind("mpi", 2),
    )
    return locals()


if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=eval_rm.main, mode="local"
        )
    )
