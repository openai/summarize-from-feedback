#!/usr/bin/env python3

import fire
import os
import pandas as pd
import numpy as np
import json

from summarize_from_feedback import eval_refinements
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher
from summarize_from_feedback.tasks import TaskResponseHParams, TaskHParams, TaskQueryHParams
import summarize_from_feedback
from summarize_from_feedback import tasks
from summarize_from_feedback.datasets.jsonl_encoding import encode_example

def experiment_definitions():
    reward_model_spec = combos(
        bind("device", "cpu"),
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4"),
        bind("short_name", "rm4"),
    )
    tldr_task = combos(
    bind(
        "query.format_str", "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    ),
    bind("query.dataset", "tldr_3_filtered"),
    bind("query.length", 512),
    bind("response.length", 48),
    bind("query.truncate_text", "\n"),
    bind("query.truncate_field", "post"),
    bind("query.pad_side", "left"),
    bind("response.truncate_token", 50256),  # endoftext
    bind("response.ref_format_str", " {reference}"),  # add a leading space
)

    reward_model_cpu = combos(
        bind_nested("task", tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", reward_model_spec),
        bind("fp16_activations", False)
    )
    return locals()


def prepare_results():
    results_folder = "/Users/jeremyscheurer/Code/language_alignment/language_feedback_learning/data/results/tldr_summarization"
    prompt_types = ["summary", "refinement", "summary_feedback", "summary_refinement_summary", "summary_refinement_summary_generate_summary", "summary_refinement_summary_generate_refinement", "summary_feedback_summary_feedback", "summary_feedback_refinement_summary_feedback", "summary_feedback_refinement_generate_summary", "summary_feedback_refinement_generate_refinement"]
    models = ["ada", "babbage", "curie", "davinci"]
    context_format = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    number_of_generated_samples = 5
    task_response_hparams = TaskResponseHParams()
    task_response_hparams.length = 48
    task_response_hparams.ref_format_str = " {reference}"
    task_response_hparams.truncate_token = 50256

    task_query_hparams = TaskQueryHParams()
    task_query_hparams.format_str = context_format
    task_query_hparams.length = 512
    task_query_hparams.truncate_text = "\n"
    task_query_hparams.truncate_field = "post"
    task_query_hparams.pad_side = "left"
    task_query_hparams.padding = None

    task_hparams = TaskHParams
    task_hparams.response = task_response_hparams
    task_hparams.query = task_query_hparams

    for prompt_type in prompt_types:
        for model in models:
            results_file = os.path.join(results_folder, "{}_{}_tldr_summarization_results_2_shot.json".format(model, prompt_type))
            if os.path.isfile(results_file):
                reformated_results_list = []
                results_df = pd.read_json(results_file)
                number_of_samples = results_df.shape[0]
                reformated_result = {}
                for sample_id in range(number_of_samples):
                    results_sample = results_df.iloc[sample_id]
                    context_text = context_format.format(subreddit=results_sample["subreddit"],
                                          title=results_sample["title"],
                                          post=results_sample["post"])
                    reformated_result["context"] = context_text
                    samples = []
                    for i in range(number_of_generated_samples):
                        samples.append(results_sample["predicted_refinement_{}".format(i)])

                    reformated_result["samples"] = samples
                    reformated_result["original_summary"] = results_sample["text"]
                    reformated_result["target"] = results_sample["target"]
                    extra_fields = {}
                    extra_fields["id"] = results_sample["id"]
                    extra_fields["subreddit"] = results_sample["subreddit"]
                    extra_fields["title"] = results_sample["title"]
                    extra_fields["post"] = results_sample["post"]

                    reformated_result["extra_fields"] = extra_fields

                    response_encoder = tasks.ResponseEncoder(task_response_hparams, summarize_from_feedback.encoder)
                    query_data_fields = {"subreddit":results_sample["subreddit"],
                                          "title":results_sample["title"],
                                          "post":results_sample["post"]}
                    query_info = tasks.process_query(query_data_fields, encoder=summarize_from_feedback.encoder, hparams=task_hparams.query)

                    all_sample_tokens = []
                    for i in range(number_of_generated_samples):
                        sample_tokens = response_encoder.encode_response(reformated_result["samples"][i], allow_truncate=True)
                        all_sample_tokens.append(sample_tokens)

                    original_summary_tokens = response_encoder.encode_response(reformated_result["original_summary"], allow_truncate=True)
                    target_tokens = response_encoder.encode_response(reformated_result["target"], allow_truncate=True)

                    reformated_result["context_tokens"] = np.array(query_info["tokens"])
                    reformated_result["sample_tokens"] = np.array(all_sample_tokens)
                    reformated_result["original_summary_tokens"] = np.array([original_summary_tokens])
                    reformated_result["target_tokens"] = np.array([target_tokens])


                    reformated_results_list.append(encode_example(reformated_result))
                with open(os.path.join(results_folder, "reformated_results/samples.{}_{}_tldr_results.jsonl".format(model, prompt_type)), "w") as outfile:
                    for entry in reformated_results_list:
                        json.dump(entry, outfile)
                        outfile.write('\n')

if __name__ == "__main__":
    prepare_results()
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=eval_refinements.main, mode="local"
        )
    )
