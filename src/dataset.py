"""
Custom Dataset for Semantic Front-End Training
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SemanticDataset(Dataset):
    """Custom dataset for semantic front-end training."""

    def __init__(self, data, tokenizer_model="distilbert-base-uncased"):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, coords = self.data[idx]
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        labels = torch.tensor(coords, dtype=torch.float32) * 2
        return inputs, labels
