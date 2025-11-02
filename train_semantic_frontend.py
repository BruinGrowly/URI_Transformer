"""
Training Script for the Semantic Front-End
"""

import torch
from torch import nn, optim
from src.semantic_frontend import SemanticFrontEnd
from src.training_data import TRAINING_DATA
import random
from torch.utils.data import DataLoader, TensorDataset

# Configuration
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 8
MODEL_SAVE_PATH = "trained_semantic_frontend_model.pth"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def train():
    """Trains the ProjectionHead."""
    print("--- Training Semantic Front-End ---")

    # Prepare data
    sentences = [item[0] for item in TRAINING_DATA]
    labels = torch.tensor([item[1] for item in TRAINING_DATA], dtype=torch.float32) * 2

    # Initialize model
    frontend = SemanticFrontEnd()

    # Create DataLoader
    with torch.no_grad():
        inputs = frontend.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
        embeddings = frontend.language_model(**inputs).last_hidden_state[:, 0, :]

    dataset = TensorDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.AdamW(frontend.projection_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        frontend.projection_head.train()
        total_loss = 0
        for batch_embeddings, batch_labels in loader:
            optimizer.zero_grad()
            predictions = frontend.projection_head(batch_embeddings)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss / len(loader):.4f}")

    # Evaluate
    frontend.projection_head.eval()
    with torch.no_grad():
        predictions = frontend.projection_head(embeddings)
        final_loss = criterion(predictions, labels).item()
        ss_res = torch.sum((labels - predictions) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels, dim=0)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final R² Score: {r2_score:.4f}")

    # Save model
    torch.save(frontend.projection_head.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
