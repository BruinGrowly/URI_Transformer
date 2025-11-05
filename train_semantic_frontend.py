"""
Training Script for the Semantic Front-End
"""

import torch
from torch import nn, optim
from src.semantic_frontend import SemanticFrontEnd
from src.training_data import TRAINING_DATA
from src.dataset import SemanticDataset
import random
from torch.utils.data import DataLoader

# Configuration
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 8
MODEL_SAVE_PATH = "trained_semantic_frontend_model.pth"
RANDOM_SEED = 42

def train():
    """Trains the ProjectionHead."""
    print("--- Training Semantic Front-End ---")

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Initialize model and dataset
    frontend = SemanticFrontEnd()
    dataset = SemanticDataset(TRAINING_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.AdamW(frontend.projection_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        frontend.projection_head.train()
        total_loss = 0
        for inputs, labels in loader:
            # Move inputs to the correct device
            inputs = {key: val.to(frontend.language_model.device) for key, val in inputs.items()}
            labels = labels.to(frontend.language_model.device)

            # Get embeddings from the language model
            with torch.no_grad():
                embeddings = frontend.language_model(input_ids=inputs['input_ids'].squeeze(1), attention_mask=inputs['attention_mask'].squeeze(1)).last_hidden_state[:, 0, :]

            optimizer.zero_grad()
            predictions = frontend.projection_head(embeddings)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss / len(loader):.4f}")

    # Evaluate
    frontend.projection_head.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {key: val.to(frontend.language_model.device) for key, val in inputs.items()}
            embeddings = frontend.language_model(**inputs).last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    with torch.no_grad():
        predictions = frontend.projection_head(all_embeddings)
        final_loss = criterion(predictions, all_labels).item()
        ss_res = torch.sum((all_labels - predictions) ** 2)
        ss_tot = torch.sum((all_labels - torch.mean(all_labels, dim=0)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final R² Score: {r2_score:.4f}")

    # Save model
    torch.save(frontend.projection_head.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
