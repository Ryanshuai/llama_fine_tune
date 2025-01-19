from typing import List, Dict

from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.examples = self._preprocess_texts(texts)

    def _preprocess_texts(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        examples = []
        for text in texts:
            tokenized = self.tokenizer(text, truncation=False, return_tensors="pt")
            input_ids = tokenized['input_ids'][0]

            for i in range(0, len(input_ids), self.stride):
                chunk = input_ids[i:i + self.max_length]
                if len(chunk) < self.max_length // 2:
                    continue
                examples.append({'input_ids': chunk, 'labels': chunk.clone()})

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
