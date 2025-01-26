import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from data.alpaca_dataset import AlpacaDataset
from data.collator import AlpacaDataCollator
from model import model, tokenizer


class Trainer:
    def __init__(self):
        self.output_dir = "./checkpoints"
        self.num_epochs = 3
        self.batch_size = 4
        self.gradient_accumulation_steps = 8
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.eval_steps = 100
        self.max_grad_norm = 1.0

        self.model = model
        self.tokenizer = tokenizer

        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = AlpacaDataset("data/all_instances_82K.jsonl", self.tokenizer)
        self.data_collator = AlpacaDataCollator(tokenizer=self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     collate_fn=self.data_collator)

        self.optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        self.step = 0

        max_train_steps = len(self.dataloader) // self.gradient_accumulation_steps * self.num_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=max_train_steps * self.warmup_ratio,
            num_training_steps=max_train_steps,
        )

    def train_step(self, batch):
        self.model.train()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.amp.autocast('cuda'):
            outputs = model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        self.step += 1
        return loss.item()

    def train_epoch(self):
        for batch in tqdm(self.dataloader):
            self.train_step(batch)

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch()

            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch}")
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
