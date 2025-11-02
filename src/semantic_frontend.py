"""
Hybrid Semantic Front-End
"""

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from src.phi_geometric_engine import PhiCoordinate

class ProjectionHead(nn.Module):
    """
    A simple neural network that projects a high-dimensional vector
    into our 4D PhiCoordinate space. It uses a single linear layer
    to directly map embeddings to coordinates.
    """
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        x = self.sigmoid(x) * 2  # Scale output to [0, 2]
        return x

class SemanticFrontEnd:
    """
    A hybrid semantic front-end that uses a pre-trained language model
    and a trained projection head to map text to a PhiCoordinate.
    """
    def __init__(
        self,
        model_path="distilbert-base-uncased",
        projection_head_path=None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.language_model = AutoModel.from_pretrained(model_path)

        input_dim = self.language_model.config.hidden_size
        self.projection_head = ProjectionHead(input_dim=input_dim)

        if projection_head_path:
            self.projection_head.load_state_dict(
                torch.load(projection_head_path)
            )

        self.projection_head.eval()  # Set to evaluation mode by default

    def text_to_coordinate(self, text: str) -> PhiCoordinate:
        """
        Analyzes raw text with the hybrid model to generate a PhiCoordinate.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.language_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        with torch.no_grad():
            phi_vector = self.projection_head(cls_embedding)

        coords = phi_vector.squeeze().tolist()
        return PhiCoordinate(
            love=coords[0],
            justice=coords[1],
            power=coords[2],
            wisdom=coords[3]
        )
