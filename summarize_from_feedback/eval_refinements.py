import json
import os
from dataclasses import dataclass, field

import blobfile as bf
import numpy as np
import torch

from summarize_from_feedback.datasets import jsonl_encoding
from summarize_from_feedback.query_response_model import ModelSpec
from summarize_from_feedback.reward_model import RewardModel
from summarize_from_feedback.task_data import make_jsonl_samples_iter
from summarize_from_feedback.tasks import TaskHParams
from summarize_from_feedback.utils import Timer, hyperparams
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq
from summarize_from_feedback.utils.logging_utils import setup_logging_with_pacific_tz
from summarize_from_feedback.utils.torch_utils import to_numpy

"""
Evaluates a reward model on a set of query-responses examples. The output will contain the same
json data as the input along with an extra key containing the predicted reward.
"""


@dataclass
class HParams(hyperparams.HParams):
    reward_model_spec: ModelSpec = field(default_factory=ModelSpec)
    task: TaskHParams = field(default_factory=TaskHParams)
    input_path_folder: str = None
    input_path_index: int = None
    output_folder: str = None
    fp16_activations: bool = True
    output_key: str = "predicted_reward"

all_input_paths = ['samples.curie_summary_feedback_refinement_generate_refinement_tldr_results.jsonl', 'samples.davinci_summary_refinement_summary_tldr_results.jsonl', 'samples.ada_summary_refinement_summary_generate_summary_tldr_results.jsonl', 'samples.ada_summary_refinement_summary_generate_refinement_tldr_results.jsonl', 'samples.babbage_summary_feedback_refinement_summary_feedback_tldr_results.jsonl', 'samples.ada_refinement_tldr_results.jsonl', 'samples.davinci_summary_feedback_refinement_generate_refinement_tldr_results.jsonl', 'samples.babbage_summary_feedback_refinement_generate_summary_tldr_results.jsonl', 'samples.curie_summary_refinement_summary_generate_summary_tldr_results.jsonl', 'samples.davinci_summary_refinement_summary_generate_refinement_tldr_results.jsonl', 'samples.babbage_summary_refinement_summary_generate_summary_tldr_results.jsonl', 'samples.babbage_summary_refinement_summary_tldr_results.jsonl', 'samples.ada_summary_feedback_refinement_generate_refinement_tldr_results.jsonl', 'samples.babbage_summary_tldr_results.jsonl', 'samples.davinci_summary_feedback_refinement_summary_feedback_tldr_results.jsonl', 'samples.curie_summary_tldr_results.jsonl', 'samples.babbage_summary_feedback_refinement_generate_refinement_tldr_results.jsonl', 'samples.ada_summary_refinement_summary_tldr_results.jsonl', 'samples.davinci_summary_refinement_summary_generate_summary_tldr_results.jsonl', 'samples.curie_refinement_tldr_results.jsonl', 'samples.ada_summary_feedback_refinement_summary_feedback_tldr_results.jsonl', 'samples.curie_summary_feedback_refinement_generate_summary_tldr_results.jsonl', 'samples.davinci_summary_tldr_results.jsonl', 'samples.curie_summary_feedback_refinement_summary_feedback_tldr_results.jsonl', 'samples.ada_summary_feedback_refinement_generate_summary_tldr_results.jsonl', 'samples.babbage_summary_refinement_summary_generate_refinement_tldr_results.jsonl', 'samples.ada_summary_tldr_results.jsonl', 'samples.babbage_refinement_tldr_results.jsonl', 'samples.curie_summary_refinement_summary_generate_refinement_tldr_results.jsonl', 'samples.curie_summary_refinement_summary_tldr_results.jsonl', 'samples.davinci_refinement_tldr_results.jsonl', 'samples.davinci_summary_feedback_refinement_generate_summary_tldr_results.jsonl']


def main(H: HParams):
    H.input_path = os.path.join(H.input_path_folder, all_input_paths[H.input_path_index])
    assert os.path.isfile(H.input_path), H.input_path
    layout = H.reward_model_spec.run_params.all_gpu_layout()

    reward_model = RewardModel(task_hparams=H.task, spec=H.reward_model_spec, layout=layout)

    setup_logging_with_pacific_tz()

    act_dtype = torch.float16 if H.fp16_activations else torch.float32

    results_dir = H.output_folder
    bf.makedirs(results_dir)


    # Creates files for printing. Only the replica root prints the files

    experiment_name = os.path.split(H.input_path)[1].split(".")[1] + ".jsonl"
    if not os.path.isdir(H.output_folder):
        os.mkdir(H.output_folder)
    output_file_name = os.path.join(H.output_folder, experiment_name)
    print(f"Outputs will be written to {output_file_name}")

    if layout.is_logging_rank:
        with open(os.path.join(results_dir, experiment_name +"task_hparams.json"), "w") as f:
            json.dump(H.task.to_json(), f)
        with open(os.path.join(results_dir, experiment_name +"hparams.json"), "w") as f:
            json.dump(H.to_json(), f)

    input_iter = make_jsonl_samples_iter(H.input_path, layout=layout)

    replica_rewards = []
    replica_original_summary_rewards = []
    replica_target_rewards = []

    with open(output_file_name, "a") as out_f:
        input_idx = 0
        for input in input_iter:
            with Timer() as timer:
                query_tokens = torch.tensor(input["context_tokens"])
                assert_shape_eq(
                    query_tokens, (H.task.query.length,), "Context tokens shape mismatch"
                )
                response_tokens = torch.tensor(input["sample_tokens"])
                assert_eq(response_tokens.dim(), 2)

                n_responses = response_tokens.size(0)

                results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=response_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                rewards = to_numpy(results["reward"].reshape((n_responses,)))

                original_summary_tokens = torch.tensor(input["original_summary_tokens"])

                original_summary_results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=original_summary_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                original_summary_reward = to_numpy(original_summary_results["reward"])

                target_tokens = torch.tensor(input["target_tokens"])
                target_results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=target_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                target_reward = to_numpy(target_results["reward"])


                if layout.is_replica_root:
                    replica_rewards.append(rewards)
                    replica_original_summary_rewards.append(original_summary_reward)
                    replica_target_rewards.append(target_reward)

                    output = {**input, H.output_key: rewards, "original_summary_reward": original_summary_reward, "target_reward": target_reward}
                    out_f.write((json.dumps(jsonl_encoding.encode_example(output)) + "\n"))
            input_idx += 1
            if layout.is_replica_root:
                print(f"Batch {input_idx}.  Took {timer.interval} seconds")

        if layout.is_replica_root:
            print(f"Wrote {input_idx} batches to {output_file_name}")

            replica_rewards = np.stack(replica_rewards, axis=0)
            replica_original_summary_rewards = np.stack(replica_original_summary_rewards, axis=0)
            replica_target_rewards = np.stack(replica_target_rewards, axis=0)

            all_rewards = reward_model.dp_comm.mpi_all_gather(replica_rewards, "rewards")
            all_original_summary_rewards = reward_model.dp_comm.mpi_all_gather(replica_original_summary_rewards, "original_summary_rewards")
            all_target_rewards = reward_model.dp_comm.mpi_all_gather(replica_target_rewards, "target_rewards")

            if layout.replica_idx == 0:
                all_rewards = np.concatenate(all_rewards, axis=0)
                all_original_summary_rewards = np.concatenate(all_original_summary_rewards, axis=0)
                all_target_rewards = np.concatenate(all_target_rewards, axis=0)

                print(f"Mean predicted reward: {all_rewards.mean():.3f}")
                if all_rewards.shape[1] > 1:
                    print(f"Stddev within a query: {all_rewards.std(axis=1, ddof=1).mean():.3}")
                print(f"Stddev across queries: {all_rewards.std(axis=0, ddof=1).mean():.3}")

                print("------")
                print(f"Mean original summary reward: {all_original_summary_rewards.mean():.3f}")
                print(f"Stddev across queries: {all_original_summary_rewards.std(axis=0, ddof=1).mean():.3}")

                print("-------")
                print(f"Mean target reward: {all_target_rewards.mean():.3f}")
                print(f"Stddev across queries: {all_target_rewards.std(axis=0, ddof=1).mean():.3}")

    return dict(output_path=results_dir)
