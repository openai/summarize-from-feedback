import json
import os
from functools import partial
from glob import glob
from typing import Optional

import blobfile as bf
import torch

import summarize_from_feedback
from summarize_from_feedback import tasks
from summarize_from_feedback.datasets import jsonl_encoding, get_dataset
from summarize_from_feedback.model_layout import ModelLayout
from summarize_from_feedback.utils import even_more_itertools, blobs


def _collate_fn(raw_data, all_fields=False, device="cpu"):
    context_input = torch.as_tensor(
        [x["context"]["tokens"] for x in raw_data], dtype=torch.long, device=device
    )
    reference_input = torch.as_tensor(
        [x["reference"]["tokens"] for x in raw_data], dtype=torch.long, device=device
    )
    input_dict = dict(context=dict(tokens=context_input), reference=dict(tokens=reference_input))
    if "text" in raw_data[0]["reference"]:
        input_dict["reference"]["text"] = [x["reference"]["text"] for x in raw_data]
    if all_fields:
        input_dict["extra_fields"] = [x["extra_fields"] for x in raw_data]
    return input_dict


class _DataLoaderWrapper(torch.utils.data.IterableDataset):
    """
    torch.utils.data.DataLoader behaves differently depending on the class of the iterator it is passed.
    This wrapper lets us use the iterable setup.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


def torch_loader(iterable, batch_size, num_workers=1, drop_last=False, collate_fn=None):
    assert num_workers in (0, 1)
    loader = torch.utils.data.DataLoader(
        _DataLoaderWrapper(iterable),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return iter(loader)


def get_iter_for_task(
    task_H,
    *,
    encoder=summarize_from_feedback.encoder,
    dataset_split,
    batch_size,
    seed,
    layout: Optional[ModelLayout] = None,
    repeat=True,
    all_fields=False,
):
    response_encoder = tasks.ResponseEncoder(task_H.response, encoder)

    def map_input(raw_data):
        print(raw_data)
        ref_response = task_H.response.ref_format_str.format(**raw_data)
        ref_tokens = response_encoder.encode_response(ref_response, allow_truncate=True)
        query_info = tasks.process_query(raw_data, encoder=encoder, hparams=task_H.query)
        return dict(
            context=query_info,
            # NOTE: tokens are truncated but text is not
            reference=dict(tokens=ref_tokens, text=ref_response),
            # NOTE: we remove reference to prevent mistakes, after the rm4 space bug
            extra_fields={k: v for k, v in raw_data.items() if k != "reference"}
            if all_fields
            else dict(),
        )

    ds = get_dataset(
        task_H.query.dataset, split=dataset_split, seed=seed, repeat=repeat, layout=layout
    )
    ds = map(map_input, ds)
    ds = torch_loader(
        ds,
        num_workers=1,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=partial(_collate_fn, all_fields=all_fields),
    )
    return ds


def make_jsonl_samples_iter(input_path, layout: Optional[ModelLayout] = None):
    """
    Makes an iterator reading examples out of all the samples.[0-9]*.jsonl files in the given path,
    distributed across replicas according to the layout.
    """
    local_input_path = input_path
    assert os.path.isfile(local_input_path)
    print("Looking for samples in folder {}".format(local_input_path))

    def all_examples():
        with open(local_input_path, "r") as f:
            for line in f:
                encoded_example = json.loads(line)
                example = jsonl_encoding.decode_example(encoded_example)
                yield example

    d = all_examples()
    if layout:
        d = even_more_itertools.distribute(d, layout)
    return d
