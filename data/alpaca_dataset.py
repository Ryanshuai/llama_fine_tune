import copy
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Sequence

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class AlpacaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer):
        super().__init__()
        list_data_dict = []
        with open(data_path, 'r') as f:
            for line in f:
                list_data_dict.append(json.loads(line))

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



if __name__ == '__main__':
    from transformers import LlamaTokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

    data_path = "../resources/alpaca_sample.json"

    # tokenizer = LlamaTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    dataset = AlpacaDataset(data_path, tokenizer=tokenizer)
    data_collator = AlpacaDataCollator(tokenizer=tokenizer)

    for data in dataset:
        print(data)

    for data in data_collator([dataset[0], dataset[1]]):
        print(data)
