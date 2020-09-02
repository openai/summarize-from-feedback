import json
import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import blobfile as bf

from summarize_from_feedback import task_data
from summarize_from_feedback.datasets import jsonl_encoding
from summarize_from_feedback.utils import exact_div, Timer, hyperparams
from summarize_from_feedback.utils.logging_utils import setup_logging_with_pacific_tz
from summarize_from_feedback.utils.torch_utils import label_logprobs
from summarize_from_feedback.utils.nested import map_nested
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq
from summarize_from_feedback.query_response_model import ModelSpec, SampleHParams, PADDING_TOKEN
from summarize_from_feedback.policy import Policy
from summarize_from_feedback.tasks import TaskHParams, ResponseEncoder


@dataclass
class HParams(hyperparams.HParams):
    model_spec: ModelSpec = field(default_factory=ModelSpec)
    orig_model_spec: Optional[ModelSpec] = None
    task: TaskHParams = field(default_factory=TaskHParams)
    query_dataset_split: str = None
    sample: SampleHParams = field(default_factory=SampleHParams)
    num_queries: int = None
    # Note that these batch sizes are in # of datapoints; each datapoint may turn into multiple
    # sequences fed to the model
    queries_per_run_per_replica: int = 1
    responses_per_query: int = 1
    responses_per_query_per_batch: int = 1
    seed: int = 0
    fp16_activations: bool = True


INVALID_LOGPROB = 1.0


def avg_negative(x):
    mask = x <= 0
    return np.sum(x * mask) / np.sum(mask)


def main(H: HParams):
    layout = H.model_spec.run_params.all_gpu_layout()

    # Instantiate policy
    policy = Policy(task_hparams=H.task, spec=H.model_spec, layout=layout)

    if H.orig_model_spec:
        assert H.orig_model_spec.run_params.n_shards == H.model_spec.run_params.n_shards
        orig_policy = Policy(task_hparams=H.task, spec=H.orig_model_spec, layout=layout)
    else:
        orig_policy = None

    encoder = policy.encoder
    response_encoder = ResponseEncoder(H.task.response, encoder)
    setup_logging_with_pacific_tz()

    act_dtype = torch.float16 if H.fp16_activations else torch.float32

    is_logging_rank = layout.is_logging_rank

    total_queries_per_replica = exact_div(H.num_queries, layout.n_replicas)
    num_runs = exact_div(total_queries_per_replica, H.queries_per_run_per_replica)

    input_iter = task_data.get_iter_for_task(
        H.task,
        encoder=encoder,
        dataset_split=H.query_dataset_split,
        batch_size=H.queries_per_run_per_replica,
        layout=layout,
        seed=H.seed,
        all_fields=True,
    )

    log_dir = os.getenv("OUTPUT_DIR") or os.path.join("/tmp/jobs", os.getenv("JOB_NAME"))

    results_dir = os.path.join(log_dir, "results")
    bf.makedirs(results_dir)
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(H.to_json(), f, indent=2)

    if is_logging_rank:
        with open(os.path.join(results_dir, "task_hparams.json"), "w") as f:
            json.dump(H.task.to_json(), f)
        with open(os.path.join(results_dir, "hparams.json"), "w") as f:
            json.dump(H.to_json(), f)

    # Creates files for printing. Only the replica root prints the files
    local_file_name = os.devnull
    if layout.is_replica_root:
        fname = f"samples.{layout.replica_idx}.jsonl"
        local_file_name = os.path.join(results_dir, fname)
        print(f"Samples will be written to {local_file_name}")

    def prepare_eval_fn_and_inputs(tokens):
        def eval_fn(outputs_mb, eval_inputs_mb):
            logprobs = label_logprobs(logits=outputs_mb["logits"], labels=eval_inputs_mb["labels"])
            logprobs = torch.masked_fill(logprobs, eval_inputs_mb["mask"], INVALID_LOGPROB)
            return dict(logprobs=logprobs)

        mask = tokens == PADDING_TOKEN
        return eval_fn, dict(labels=torch.masked_fill(tokens, mask, 0), mask=mask)

    runs_per_query = exact_div(H.responses_per_query, H.responses_per_query_per_batch)
    with open(local_file_name, "w") as f:
        for run_idx in range(num_runs):
            with Timer() as timer:
                input = next(input_iter)
                context_tokens = input["context"]["tokens"]
                assert_shape_eq(
                    context_tokens,
                    (H.queries_per_run_per_replica, H.task.query.length),
                    "Context tokens shape mismatch",
                )
                ref_tokens = input["reference"]["tokens"].unsqueeze(1)
                assert_shape_eq(
                    ref_tokens,
                    (H.queries_per_run_per_replica, 1, H.task.response.length),
                    "Ref tokens shape mismatch",
                )

                # Sample from policy
                all_sample_results = []
                for _ in range(runs_per_query):
                    sample_results = policy.sample(
                        context_tokens,
                        responses_per_query=H.responses_per_query_per_batch,
                        sample_H=H.sample,
                        act_dtype=act_dtype,
                    )
                    assert_shape_eq(
                        sample_results["samples"],
                        (
                            H.queries_per_run_per_replica,
                            H.responses_per_query_per_batch,
                            H.task.response.length,
                        ),
                        "Samples size mismatch",
                    )

                    processed_samples = response_encoder.process_responses(
                        sample_results["samples"]
                    )

                    sample_results["processed_samples"] = processed_samples
                    assert_shape_eq(
                        processed_samples,
                        (
                            H.queries_per_run_per_replica,
                            H.responses_per_query_per_batch,
                            H.task.response.length,
                        ),
                        "Samples size mismatch",
                    )
                    sample_results["logprobs"] = torch.masked_fill(
                        sample_results["logprobs"],
                        processed_samples == PADDING_TOKEN,
                        INVALID_LOGPROB,
                    )
                    if orig_policy is not None:
                        eval_fn, eval_inputs = prepare_eval_fn_and_inputs(processed_samples)
                        orig_eval_results = orig_policy.eval(
                            context_tokens,
                            processed_samples,
                            eval_fn=eval_fn,
                            eval_inputs=eval_inputs,
                            act_dtype=act_dtype,
                        )
                        sample_results["orig_eval_results"] = orig_eval_results

                    sample_results = map_nested(sample_results, lambda x: x.cpu().numpy())
                    all_sample_results.append(sample_results)

                eval_fn, eval_inputs = prepare_eval_fn_and_inputs(ref_tokens)
                ref_eval_results = policy.eval(
                    context_tokens,
                    ref_tokens,
                    eval_fn=eval_fn,
                    eval_inputs=eval_inputs,
                    act_dtype=act_dtype,
                )

                if orig_policy is not None:
                    orig_ref_eval_results = orig_policy.eval(
                        context_tokens,
                        ref_tokens,
                        eval_fn=eval_fn,
                        eval_inputs=eval_inputs,
                        act_dtype=act_dtype,
                    )

                if layout.is_replica_root:
                    for batch_idx in range(H.queries_per_run_per_replica):
                        context_tokens = sample_results["contexts"][batch_idx]
                        context = encoder.decode(context_tokens)

                        # Dump to a file so that we can use things in downstream tasks
                        # samples (written to file) is now a list of strings
                        d = dict(context=context, context_tokens=context_tokens)
                        d["sample_tokens"] = np.concatenate(
                            [
                                sample_results["processed_samples"][batch_idx]
                                for sample_results in all_sample_results
                            ],
                            axis=0,
                        )
                        assert_shape_eq(
                            d["sample_tokens"],
                            (H.responses_per_query, H.task.response.length),
                            "Sample tokens shape mismatch",
                        )
                        d["samples"] = response_encoder.decode_responses(d["sample_tokens"])
                        d["logprobs"] = np.concatenate(
                            [
                                sample_results["logprobs"][batch_idx]
                                for sample_results in all_sample_results
                            ],
                            axis=0,
                        )
                        assert_shape_eq(
                            d["logprobs"],
                            (H.responses_per_query, H.task.response.length),
                            "Logprobs shape mismatch",
                        )
                        if orig_policy is not None:
                            d["orig_logprobs"] = np.concatenate(
                                [
                                    sample_results["orig_eval_results"]["eval_stats"]["logprobs"][
                                        batch_idx
                                    ]
                                    for sample_results in all_sample_results
                                ],
                                axis=0,
                            )
                            assert_shape_eq(
                                d["orig_logprobs"],
                                (H.responses_per_query, H.task.response.length),
                                "Orig logprobs shape mismatch",
                            )

                        # Process ref_tokens (H.queries_per_run_per_replica, H.task.response.length)
                        d["ref_tokens"] = ref_tokens[batch_idx].squeeze(0).cpu().numpy()
                        d["ref"] = response_encoder.decode_response(d["ref_tokens"])
                        assert_eq(
                            len(d["ref_tokens"]),
                            H.task.response.length,
                            "Ref tokens shape mismatch",
                        )
                        d["ref_logprobs"] = (
                            ref_eval_results["eval_stats"]["logprobs"][batch_idx]
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                        assert_eq(
                            len(d["ref_logprobs"]),
                            H.task.response.length,
                            "Ref logprobs shape mismatch",
                        )
                        if orig_policy is not None:
                            d["orig_ref_logprobs"] = (
                                orig_ref_eval_results["eval_stats"]["logprobs"][batch_idx]
                                .squeeze(0)
                                .cpu()
                                .numpy()
                            )
                            assert_eq(
                                len(d["orig_ref_logprobs"]),
                                H.task.response.length,
                                "Orig ref Logprobs shape mismatch",
                            )
                        if "extra_fields" in input:
                            d["extra_fields"] = input["extra_fields"][batch_idx]

                        print("=" * 80)
                        replica_sample_idx = run_idx * H.queries_per_run_per_replica + batch_idx
                        print(f"RESULT {replica_sample_idx} of {total_queries_per_replica}")
                        print(f"CONTEXT:")
                        print(context)
                        print(f"REF:")
                        print(d["ref"])
                        print("avg logprob", avg_negative(d["ref_logprobs"]))
                        if orig_policy is not None:
                            print("avg orig logprob", avg_negative(d["orig_ref_logprobs"]))
                        for sample_idx in range(H.responses_per_query):
                            print(f"SAMPLE {sample_idx}:")
                            print(d["samples"][sample_idx])
                            print("avg logprob", avg_negative(d["logprobs"][sample_idx]))
                            if orig_policy is not None:
                                print(
                                    "avg orig logprob", avg_negative(d["orig_logprobs"][sample_idx])
                                )

                        f.write((json.dumps(jsonl_encoding.encode_example(d)) + "\n"))
            if layout.is_replica_root:
                print(f"Batch {run_idx+1} of {num_runs}.  Took {timer.interval} seconds")

    return dict(output_path=results_dir)
