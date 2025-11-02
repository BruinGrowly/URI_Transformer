"""
Training Script for the Semantic Front-End
==========================================

This script trains the ProjectionHead of our hybrid semantic front-end.
It loads the labeled training data, initializes the SemanticFrontEnd,
runs a training loop with validation, and saves the best model.

Features:
- Train/Validation/Test splits (70/15/15)
- Comprehensive evaluation metrics (MAE, MSE, R²)
- Model checkpointing (saves best validation model)
- Early stopping to prevent overfitting
- Detailed logging and progress tracking
"""

import torch
from torch import nn, optim
from src.semantic_frontend import SemanticFrontEnd
from src.training_data import TRAINING_DATA
import random

# --- Configuration ---
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MODEL_SAVE_PATH = "trained_semantic_frontend_model.pth"
EARLY_STOPPING_PATIENCE = 20
RANDOM_SEED = 42
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping threshold
USE_LR_SCHEDULER = True     # Whether to use learning rate scheduling

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def split_data(data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits data into train, validation, and test sets.

    Args:
        data: List of (sentence, coordinates) tuples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)

    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]

    return train_data, val_data, test_data


def calculate_metrics(predictions, labels):
    """
    Calculates comprehensive evaluation metrics.

    Args:
        predictions: Model predictions (batch_size, 4)
        labels: Ground truth labels (batch_size, 4)

    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = nn.MSELoss()(predictions, labels).item()

    # Mean Absolute Error
    mae = nn.L1Loss()(predictions, labels).item()

    # Per-dimension MAE
    per_dim_mae = torch.mean(torch.abs(predictions - labels), dim=0)

    # R² score (coefficient of determination)
    ss_res = torch.sum((labels - predictions) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels, dim=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    # Cosine similarity (semantic alignment)
    cos_sim = nn.CosineSimilarity(dim=1)(predictions, labels).mean().item()

    return {
        'mse': mse,
        'mae': mae,
        'mae_love': per_dim_mae[0].item(),
        'mae_justice': per_dim_mae[1].item(),
        'mae_power': per_dim_mae[2].item(),
        'mae_wisdom': per_dim_mae[3].item(),
        'r2': r2,
        'cosine_similarity': cos_sim
    }


def evaluate(frontend, data, criterion):
    """
    Evaluates the model on a dataset.

    Args:
        frontend: SemanticFrontEnd instance
        data: List of (sentence, coordinates) tuples
        criterion: Loss function

    Returns:
        Dictionary of metrics
    """
    frontend.projection_head.eval()

    sentences = [item[0] for item in data]
    labels = torch.tensor([item[1] for item in data], dtype=torch.float32)

    with torch.no_grad():
        # Get embeddings from the language model
        inputs = frontend.tokenizer(
            sentences, return_tensors="pt", truncation=True, padding=True
        )
        outputs = frontend.language_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Forward pass through projection head
        predictions = frontend.projection_head(embeddings)

        # Calculate metrics
        metrics = calculate_metrics(predictions, labels)
        metrics['loss'] = criterion(predictions, labels).item()

    return metrics


def train():
    """Trains the ProjectionHead with validation and comprehensive metrics."""

    print("=" * 70)
    print("Training Semantic Front-End with Enhanced Validation")
    print("=" * 70)

    # 1. Split the data
    train_data, val_data, test_data = split_data(
        TRAINING_DATA, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
    )

    print(f"\nDataset Split:")
    print(f"  Total examples:      {len(TRAINING_DATA)}")
    print(f"  Training set:        {len(train_data)} ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation set:      {len(val_data)} ({VAL_SPLIT*100:.0f}%)")
    print(f"  Test set:            {len(test_data)} ({TEST_SPLIT*100:.0f}%)")

    # 2. Initialize the model
    print(f"\nInitializing model...")
    frontend = SemanticFrontEnd()

    # 3. Prepare training data
    train_sentences = [item[0] for item in train_data]
    train_labels = torch.tensor([item[1] for item in train_data], dtype=torch.float32)

    # 4. Define optimizer and loss function
    optimizer = optim.Adam(frontend.projection_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5. Learning rate scheduler (optional)
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    # 6. Training loop with validation
    print(f"\nTraining Configuration:")
    print(f"  Epochs:              {NUM_EPOCHS}")
    print(f"  Learning rate:       {LEARNING_RATE}")
    print(f"  Gradient clipping:   {GRADIENT_CLIP_VALUE}")
    print(f"  LR scheduling:       {USE_LR_SCHEDULER}")
    print(f"  Early stopping:      {EARLY_STOPPING_PATIENCE} epochs")
    print(f"  Random seed:         {RANDOM_SEED}")

    print(f"\n{'='*70}")
    print(f"Starting Training...")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        # Training phase
        frontend.projection_head.train()

        # Get embeddings from the language model
        inputs = frontend.tokenizer(
            train_sentences, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = frontend.language_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the projection head
        predictions = frontend.projection_head(embeddings)

        # Calculate the loss
        train_loss = criterion(predictions, train_labels)

        # Backward pass and optimization
        train_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            frontend.projection_head.parameters(), GRADIENT_CLIP_VALUE
        )

        optimizer.step()

        # Validation phase
        val_metrics = evaluate(frontend, val_data, criterion)
        val_loss = val_metrics['loss']

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = frontend.projection_head.state_dict().copy()
        else:
            epochs_without_improvement += 1

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.4f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"No Improve: {epochs_without_improvement}")

        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    # 6. Load best model
    frontend.projection_head.load_state_dict(best_model_state)

    # 7. Final evaluation on all sets
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}\n")

    train_metrics = evaluate(frontend, train_data, criterion)
    val_metrics = evaluate(frontend, val_data, criterion)
    test_metrics = evaluate(frontend, test_data, criterion)

    print("Training Set:")
    print(f"  Loss:              {train_metrics['loss']:.4f}")
    print(f"  MAE:               {train_metrics['mae']:.4f}")
    print(f"  MSE:               {train_metrics['mse']:.4f}")
    print(f"  R²:                {train_metrics['r2']:.4f}")
    print(f"  Cosine Similarity: {train_metrics['cosine_similarity']:.4f}")
    print(f"  MAE (Love):        {train_metrics['mae_love']:.4f}")
    print(f"  MAE (Justice):     {train_metrics['mae_justice']:.4f}")
    print(f"  MAE (Power):       {train_metrics['mae_power']:.4f}")
    print(f"  MAE (Wisdom):      {train_metrics['mae_wisdom']:.4f}")

    print("\nValidation Set:")
    print(f"  Loss:              {val_metrics['loss']:.4f}")
    print(f"  MAE:               {val_metrics['mae']:.4f}")
    print(f"  MSE:               {val_metrics['mse']:.4f}")
    print(f"  R²:                {val_metrics['r2']:.4f}")
    print(f"  Cosine Similarity: {val_metrics['cosine_similarity']:.4f}")
    print(f"  MAE (Love):        {val_metrics['mae_love']:.4f}")
    print(f"  MAE (Justice):     {val_metrics['mae_justice']:.4f}")
    print(f"  MAE (Power):       {val_metrics['mae_power']:.4f}")
    print(f"  MAE (Wisdom):      {val_metrics['mae_wisdom']:.4f}")

    print("\nTest Set:")
    print(f"  Loss:              {test_metrics['loss']:.4f}")
    print(f"  MAE:               {test_metrics['mae']:.4f}")
    print(f"  MSE:               {test_metrics['mse']:.4f}")
    print(f"  R²:                {test_metrics['r2']:.4f}")
    print(f"  Cosine Similarity: {test_metrics['cosine_similarity']:.4f}")
    print(f"  MAE (Love):        {test_metrics['mae_love']:.4f}")
    print(f"  MAE (Justice):     {test_metrics['mae_justice']:.4f}")
    print(f"  MAE (Power):       {test_metrics['mae_power']:.4f}")
    print(f"  MAE (Wisdom):      {test_metrics['mae_wisdom']:.4f}")

    # 8. Save the best model
    torch.save(best_model_state, MODEL_SAVE_PATH)

    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train()
