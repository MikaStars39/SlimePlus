import abc
import copy
import logging
import os
import itertools
import ray

from pathlib import Path
from datasets import load_dataset

from slime.rollout.data_source import DataSource
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

def _check_file_type(file_path: str) -> str:
    if file_path.endswith(".jsonl") or file_path.endswith(".json"):
        return "json"
    elif file_path.endswith(".parquet"):
        return "parquet"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

@ray.remote
class StreamingRolloutDataSource(DataSource):
    def __init__(
        self,
        dataset_path: str,
        args,
        start_prompt_offset: int = 0,
        start_sample_remainder: int = 0,
        start_sample_index: int = 0,
    ):
        self.args = args
        self.dataset = load_dataset(
            _check_file_type(dataset_path), 
            data_files=dataset_path, 
            streaming=True
        )["train"]
        self.tokenizer = None
        if getattr(args, "apply_chat_template", False):
            self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

        if args.rollout_shuffle:
            self.dataset = self.dataset.shuffle(seed=args.rollout_seed, buffer_size=10000)

        self.iterator = iter(self.dataset)
        if start_prompt_offset > 0:
            # Skip already processed prompts when resuming.
            self.iterator = itertools.islice(self.iterator, start_prompt_offset, None)

        self.n_samples_per_prompt = args.n_samples_per_prompt
        self.sample_group_index = start_prompt_offset
        self.sample_index = start_sample_index
        self._resume_sample_remainder = start_sample_remainder

    def _format_prompt(self, row: dict):
        prompt = row.get(self.args.input_key)
        if not getattr(self.args, "apply_chat_template", False):
            return prompt

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        if not isinstance(messages, list):
            raise ValueError(f"Unsupported prompt type for chat template: {type(prompt)}")

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(getattr(self.args, "apply_chat_template_kwargs", None) or {}),
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        batch = list(itertools.islice(self.iterator, num_samples))
        if not batch:
            return []

        samples_groups = []
        for row in batch:
            group = []
            start_k = 0
            if self._resume_sample_remainder > 0:
                # If the previous run stopped mid-group, only emit the remaining
                # samples for the first resumed prompt.
                start_k = self._resume_sample_remainder
                self._resume_sample_remainder = 0

            for _ in range(start_k, self.n_samples_per_prompt):
                sample = Sample(
                    prompt=self._format_prompt(row),
                    label=row.get(self.args.label_key),
                )
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)

            self.sample_group_index += 1
            if group:
                samples_groups.append(group)

        return samples_groups

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        raise NotImplementedError("StreamingRolloutDataSource does not support save")

    def load(self, rollout_id=None):
        raise NotImplementedError("StreamingRolloutDataSource does not support load")

    def __len__(self) -> int:
        raise NotImplementedError("StreamingRolloutDataSource does not support __len__")