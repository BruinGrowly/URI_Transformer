"""
Training Script for the Semantic Front-End
==========================================

This script trains the ProjectionHead of our hybrid semantic front-end.
It loads the labeled training data, initializes the SemanticFrontEnd,
runs a training loop, and saves the trained model to a file.
"""

import torch
from torch import nn, optim
from src.semantic_frontend import SemanticFrontEnd
from src.training_data import TRAINING_DATA

# --- Configuration ---
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "semantic_frontend_model.pth"

def train():
    """Trains the ProjectionHead and saves the model."""

    print("--- Starting Training for Semantic Front-End ---")

    # 1. Initialize the model
    frontend = SemanticFrontEnd()
    frontend.projection_head.train() # Set to training mode

    # 2. Prepare the data
    sentences = [item[0] for item in TRAINING_DATA]
    labels = torch.tensor([item[1] for item in TRAINING_DATA], dtype=torch.float32)

    # 3. Define the optimizer and loss function
    optimizer = optim.Adam(frontend.projection_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error is a good choice for this task

    # 4. The Training Loop
    print(f"Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        # In a real-world scenario, we would batch the data.
        # For this small dataset, we can process it all at once.

        # Get embeddings from the language model
        inputs = frontend.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = frontend.language_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the projection head
        predictions = frontend.projection_head(embeddings)

        # Calculate the loss
        loss = criterion(predictions, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

    # 5. Save the trained model
    torch.save(frontend.projection_head.state_dict(), MODEL_SAVE_PATH)
    print(f"\n--- Training Complete ---")
    print(f"Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train()
