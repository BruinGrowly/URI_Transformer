"""

This module implements the new, sophisticated hybrid semantic front-end.
It uses a pre-trained language model for feature extraction and a custom
neural network "projection head" to map those features to our 4D
PhiCoordinate space.
"""

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from src.phi_geometric_engine import PhiCoordinate

class ProjectionHead(nn.Module):
    """
    A small neural network that projects a high-dimensional vector
    into our 4D PhiCoordinate space.
    """
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid() # To ensure output is in [0, 1]

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class SemanticFrontEnd:
    """
    A hybrid semantic front-end that uses a pre-trained language model
    and a trained projection head to map text to a PhiCoordinate.
    """
    def __init__(self, model_path="distilbert-base-uncased", projection_head_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.language_model = AutoModel.from_pretrained(model_path)

        # Determine the input dimension from the language model's config
        input_dim = self.language_model.config.hidden_size
        self.projection_head = ProjectionHead(input_dim=input_dim)

        if projection_head_path:
            self.projection_head.load_state_dict(torch.load(projection_head_path))

        self.projection_head.eval() # Set to evaluation mode by default

    def text_to_coordinate(self, text: str) -> PhiCoordinate:
        """
        Analyzes raw text with the hybrid model to generate a PhiCoordinate.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.language_model(**inputs)

        # Use the [CLS] token's embedding as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        with torch.no_grad():
            phi_vector = self.projection_head(cls_embedding)

        coords = phi_vector.squeeze().tolist()
        return PhiCoordinate(
            love=coords[0], justice=coords[1], power=coords[2], wisdom=coords[3]
        )
Semantic Front-End

This module provides the crucial first step in the TruthSense pipeline.
It replaces the deterministic hashing function with a lexical, meaning-driven
approach to map raw text to a 4D PhiCoordinate.
"""

from src.phi_geometric_engine import PhiCoordinate

class SemanticFrontEnd:
    """
    A lexical-based semantic front-end that maps text to a PhiCoordinate.
    """
    def __init__(self):
        self.lexical_db = self._initialize_lexical_db()

    def _initialize_lexical_db(self):
        """
        Initializes a curated lexical database.
        Values are tuples of (Love, Justice, Power, Wisdom).
        """
        return {
            # Core Concepts
            "love":       (1.0, 0.7, 0.6, 0.8),
            "justice":    (0.7, 1.0, 0.8, 0.9),
            "power":      (0.6, 0.8, 1.0, 0.7),
            "wisdom":     (0.8, 0.9, 0.7, 1.0),

            # Virtues (Love-dominant)
            "compassion": (0.9, 0.6, 0.5, 0.7),
            "kindness":   (0.9, 0.7, 0.4, 0.6),
            "humility":   (0.8, 0.8, 0.3, 0.9),

            # Principles (Justice-dominant)
            "truth":      (0.7, 0.9, 0.6, 0.9),
            "fairness":   (0.6, 0.9, 0.5, 0.7),
            "integrity":  (0.7, 0.9, 0.7, 0.8),

            # Vices (Low-Justice)
            "hate":       (0.1, 0.2, 0.8, 0.3),
            "deception":  (0.2, 0.1, 0.6, 0.4),
            "lies":       (0.3, 0.1, 0.5, 0.2),
            "unfair":     (0.4, 0.2, 0.6, 0.3),

            # Neutral/Other
            "leader":     (0.6, 0.7, 0.8, 0.8),
            "society":    (0.5, 0.6, 0.5, 0.5),
        }

    def text_to_coordinate(self, text: str) -> PhiCoordinate:
        """
        Analyzes raw text against the lexical database to generate a
        weighted PhiCoordinate.
        """
        words = text.lower().replace('.', '').replace(',', '').split()
        if not words:
            return PhiCoordinate(0.5, 0.5, 0.5, 0.5) # Neutral coordinate for empty text

        total_love, total_justice, total_power, total_wisdom = 0.0, 0.0, 0.0, 0.0
        word_count = 0

        for word in words:
            if word in self.lexical_db:
                coords = self.lexical_db[word]
                total_love += coords[0]
                total_justice += coords[1]
                total_power += coords[2]
                total_wisdom += coords[3]
                word_count += 1

        if word_count == 0:
            return PhiCoordinate(0.5, 0.5, 0.5, 0.5) # Neutral if no keywords found

        # Average the coordinates
        avg_love = total_love / word_count
        avg_justice = total_justice / word_count
        avg_power = total_power / word_count
        avg_wisdom = total_wisdom / word_count

        return PhiCoordinate(avg_love, avg_justice, avg_power, avg_wisdom)
