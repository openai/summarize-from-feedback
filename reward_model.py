from summarize_from_feedback.utils.torch_utils import to_numpy
from summarize_from_feedback.reward_model import RewardModel
from typing import Dict, Any
from functools import partial
from dataclasses import dataclass, field
from summarize_from_feedback.utils import hyperparams
from summarize_from_feedback.query_response_model import ModelSpec
import torch
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils import jobs
from summarize_from_feedback.tasks import TaskHParams, ResponseEncoder

def main():
    dict = {"query_format_string": "", "query_length": 2048, "response_length": 310}
    call_reward_model = RewardModelAPI(dict)

class RewardModelAPI:
    def __init__(self, hyperparameters: Dict[str, Any]=None):
        self.hyperparameters = self.convert_dict_to_HParams(hyperparameters)

        self.tokenizer = None

        self.layout = self.hyperparameters.reward_model_spec.run_params.all_gpu_layout()
        self.reward_model = RewardModel(task_hparams=self.hyperparameters.task, spec=self.hyperparameters.reward_model_spec, layout=self.layout)
        self.response_encoder = ResponseEncoder(self.hyperparameters.task.response, self.reward_model.encoder)

    def reward(self, context, summary):
        context_tokens = self.tokenizer.tokenize(context)
        summary_tokens = self.tokenizer.tokenize(summary)
        results = self.reward_model.reward(
            query_tokens=context_tokens,
            response_tokens=summary_tokens,
            act_dtype=torch.float32,
        )
        number_of_summaries = summary_tokens.size(0)
        rewards = to_numpy(results["reward"].reshape((number_of_summaries,)))
        return rewards



    def convert_dict_to_HParams(self, dictionary):
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

        reward_model_combo = combos(
            bind("device", "cpu"),
            bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4"),
            bind("short_name", "rm4"),
        )

        reward_model = combos(
            bind_nested("task", tldr_task),
            bind("mpi", 1),
            bind_nested("reward_model_spec", reward_model_combo),
            bind("input_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/samples/sup4_ppo_rm4"),
            bind("mode", "local")
        )
        trials =combos(
        reward_model,
        bind_nested("reward_model_spec", stub_model_spec()),
        bind("fp16_activations", False),
        )


        for trial in trials:
            descriptors = []
            trial_bindings = []
            trial_bindings_dict = {}  # only used for message

            # Extract bindings & descriptors from the trial
            for k, v, s in trial:
                if k is not None:
                    if k in trial_bindings_dict:
                        print(f"NOTE: overriding key {k} from {trial_bindings_dict[k]} to {v}")
                    trial_bindings.append((k, v))
                    trial_bindings_dict[k] = v
                if "descriptor" in s and s["descriptor"] is not "":
                    descriptors.append(str(s["descriptor"]))

            filtered_trial_bindings = []
            for k, v in trial_bindings:
                if k not in dict(mpi="mpi", mode="mode"):
                    filtered_trial_bindings.append((k, v))


            hparams = HParams()
            hparams.override_from_pairs(filtered_trial_bindings)
            return hparams

def stub_model_spec(**run_params):
    return combos(
        bind("device", "cpu"),
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/random-teeny"),
        *[bind(f"run_params.{k}", v) for (k, v) in run_params.items()],
    )

def load_model_spec(load_path, short_name=None, **run_params):
    return combos(
        bind("load_path", load_path),
        bind("short_name", short_name),
        *[bind(f"run_params.{k}", v) for (k, v) in run_params.items()],
    )
@dataclass
class HParams(hyperparams.HParams):
    reward_model_spec: ModelSpec = field(default_factory=ModelSpec)
    task: TaskHParams = field(default_factory=TaskHParams)
    input_path: str = None
    fp16_activations: bool = True
    output_key: str = "predicted_reward"

if __name__ == '__main__':
    main()