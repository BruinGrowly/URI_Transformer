# Training Guide: Semantic Front-End

This guide provides comprehensive documentation for training the URI-Transformer's Hybrid Semantic Front-End.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training Process](#training-process)
4. [Configuration](#configuration)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Overview

The Semantic Front-End training process maps high-dimensional DistilBERT embeddings (768 dimensions) to our custom 4D PhiCoordinate space representing:
- **Love** (L): Compassion, kindness, mercy
- **Justice** (J): Fairness, righteousness, truth
- **Power** (P): Authority, strength, capability
- **Wisdom** (W): Knowledge, understanding, insight

### Architecture

```
Input Text
    ↓
DistilBERT Encoder (768-dim embedding)
    ↓
Projection Head (Neural Network)
    ├── Linear(768 → 128)
    ├── ReLU activation
    ├── Linear(128 → 4)
    └── Sigmoid (output ∈ [0,1]⁴)
    ↓
PhiCoordinate (L, J, P, W)
```

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Training
```bash
python train_semantic_frontend.py
```

This will:
1. Load 362 labeled examples
2. Split into train/val/test (70/15/15)
3. Train for up to 200 epochs with early stopping
4. Save the best model to `semantic_frontend_model.pth`

### Expected Output
```
======================================================================
Training Semantic Front-End with Enhanced Validation
======================================================================

Dataset Split:
  Total examples:      362
  Training set:        362 (70%)
  Validation set:      78 (15%)
  Test set:            78 (15%)

Initializing model...

Training Configuration:
  Epochs:              200
  Learning rate:       0.001
  Early stopping:      20 epochs
  Random seed:         42

======================================================================
Starting Training...
======================================================================

Epoch [  1/200] | Train Loss: 0.0523 | Val Loss: 0.0489 | Val MAE: 0.1532 | Val R²: 0.8234 | No Improve: 0
Epoch [ 10/200] | Train Loss: 0.0234 | Val Loss: 0.0256 | Val MAE: 0.0987 | Val R²: 0.8956 | No Improve: 0
...

======================================================================
Final Evaluation
======================================================================

Training Set:
  Loss:              0.0198
  MAE:               0.0856
  MSE:               0.0198
  R²:                0.9134
  Cosine Similarity: 0.9567
  MAE (Love):        0.0823
  MAE (Justice):     0.0912
  MAE (Power):       0.0845
  MAE (Wisdom):      0.0843

Validation Set:
  Loss:              0.0256
  MAE:               0.0987
  MSE:               0.0256
  R²:                0.8956
  Cosine Similarity: 0.9423
  ...

Test Set:
  Loss:              0.0245
  MAE:               0.0945
  MSE:               0.0245
  R²:                0.8987
  Cosine Similarity: 0.9456
  ...

======================================================================
✅ Training Complete!
======================================================================
Best model saved to: semantic_frontend_model.pth
Best validation loss: 0.0256
Test MAE: 0.0945
Test R²: 0.8987
======================================================================
```

---

## Training Process

### 1. Data Loading

The training script loads data from `src/training_data.py`:
```python
TRAINING_DATA = [
    ("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8)),
    ("Justice demands equal treatment under law.", (0.5, 0.95, 0.7, 0.75)),
    # ... 516 more examples
]
```

Each example is a tuple of:
- **Sentence**: String of natural language text
- **Coordinates**: 4-tuple of floats in [0, 1] representing (L, J, P, W)

### 2. Data Splitting

Data is randomly shuffled and split:
```python
train_ratio = 0.70  # 362 examples
val_ratio   = 0.15  #  78 examples
test_ratio  = 0.15  #  78 examples
```

The split uses a fixed random seed (42) for reproducibility.

### 3. Training Loop

For each epoch:
1. **Forward Pass**: Encode sentences with DistilBERT → Project to 4D
2. **Loss Calculation**: MSE between predictions and ground truth
3. **Backward Pass**: Update projection head weights
4. **Validation**: Evaluate on validation set
5. **Checkpointing**: Save model if validation loss improved
6. **Early Stopping**: Stop if no improvement for 20 epochs

### 4. Model Selection

The training script automatically:
- Tracks best validation loss
- Saves best model state
- Loads best model at end for final evaluation

---

## Configuration

### Training Parameters

Edit `train_semantic_frontend.py` to adjust:

```python
# Hyperparameters
NUM_EPOCHS = 200                # Maximum training epochs
LEARNING_RATE = 0.001           # Adam optimizer learning rate
EARLY_STOPPING_PATIENCE = 20    # Epochs without improvement before stopping

# Data Split
TRAIN_SPLIT = 0.70              # 70% for training
VAL_SPLIT = 0.15                # 15% for validation
TEST_SPLIT = 0.15               # 15% for testing

# Reproducibility
RANDOM_SEED = 42                # For consistent splits

# Output
MODEL_SAVE_PATH = "semantic_frontend_model.pth"
```

### Model Architecture

To modify the projection head architecture, edit `src/semantic_frontend.py`:

```python
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        # Current architecture
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Example: Deeper network
        # self.fc1 = nn.Linear(input_dim, 256)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(256, 128)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(128, output_dim)
        # self.sigmoid = nn.Sigmoid()
```

---

## Evaluation Metrics

### Mean Absolute Error (MAE)
Measures average magnitude of errors:
```
MAE = (1/N) Σ |predicted_i - actual_i|
```

**Interpretation**: Lower is better. MAE of 0.1 means predictions are off by ~0.1 on average.

### Mean Squared Error (MSE)
Penalizes larger errors more heavily:
```
MSE = (1/N) Σ (predicted_i - actual_i)²
```

**Interpretation**: Lower is better. Used as the training loss function.

### R² Score (Coefficient of Determination)
Measures how well predictions match the variance in the data:
```
R² = 1 - (SS_residual / SS_total)
```

**Interpretation**:
- R² = 1.0: Perfect predictions
- R² = 0.0: As good as predicting the mean
- R² < 0.0: Worse than predicting the mean

Typical good values: R² > 0.8

### Cosine Similarity
Measures directional alignment in 4D space:
```
cos(θ) = (A · B) / (||A|| ||B||)
```

**Interpretation**:
- 1.0: Perfect alignment
- 0.0: Orthogonal
- -1.0: Opposite direction

Typical good values: > 0.9

### Per-Dimension MAE
Tracks MAE for each coordinate separately:
- **MAE (Love)**: Error on Love axis
- **MAE (Justice)**: Error on Justice axis
- **MAE (Power)**: Error on Power axis
- **MAE (Wisdom)**: Error on Wisdom axis

**Use**: Identifies which dimensions the model struggles with.

---

## Troubleshooting

### Issue: High Training Loss

**Symptoms**: Loss stays high (> 0.1) after many epochs

**Solutions**:
1. Increase training epochs: `NUM_EPOCHS = 500`
2. Decrease learning rate: `LEARNING_RATE = 0.0001`
3. Add more training data to `src/training_data.py`
4. Check data quality (ensure labels are accurate)

### Issue: Overfitting (Train << Val Loss)

**Symptoms**: Training loss much lower than validation loss

**Solutions**:
1. Reduce model complexity (smaller hidden layer)
2. Add dropout layers
3. Increase early stopping patience
4. Add more training data

### Issue: Underfitting (Both Losses High)

**Symptoms**: Both train and validation losses remain high

**Solutions**:
1. Increase model capacity (larger hidden layers)
2. Train for more epochs
3. Try different learning rates
4. Check if task is learnable (data quality)

### Issue: Poor Test Performance

**Symptoms**: Good validation metrics but poor test metrics

**Solutions**:
1. Check for data leakage between splits
2. Verify random seed is set
3. Increase dataset size
4. Check test set distribution

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Process data in smaller batches
2. Use CPU instead: `device = torch.device('cpu')`
3. Reduce model size

### Issue: Slow Training

**Symptoms**: Training takes very long

**Solutions**:
1. Use GPU if available
2. Reduce embedding computation (cache DistilBERT outputs)
3. Reduce number of epochs
4. Sample subset of data for quick experiments

---

## Advanced Usage

### Caching DistilBERT Embeddings

For faster training, pre-compute embeddings:

```python
# Create cache file
import torch
from src.semantic_frontend import SemanticFrontEnd

frontend = SemanticFrontEnd()
sentences = [item[0] for item in TRAINING_DATA]

inputs = frontend.tokenizer(sentences, return_tensors="pt",
                           truncation=True, padding=True)
with torch.no_grad():
    outputs = frontend.language_model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :]

torch.save(embeddings, "cached_embeddings.pth")
```

Then modify training script to load cached embeddings.

### Custom Loss Functions

Try alternative loss functions:

```python
# L1 Loss (MAE)
criterion = nn.L1Loss()

# Huber Loss (robust to outliers)
criterion = nn.SmoothL1Loss()

# Weighted MSE (emphasize certain dimensions)
def weighted_mse(pred, target, weights=[1.0, 1.5, 1.0, 1.2]):
    weights = torch.tensor(weights)
    return torch.mean(weights * (pred - target) ** 2)
```

### Learning Rate Scheduling

Add learning rate decay:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min',
                             factor=0.5, patience=10)

# In training loop
scheduler.step(val_loss)
```

### Cross-Validation

Implement k-fold cross-validation:

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(TRAINING_DATA)):
    print(f"Training fold {fold+1}/5")
    train_data = [TRAINING_DATA[i] for i in train_idx]
    val_data = [TRAINING_DATA[i] for i in val_idx]
    # Train model...
```

### Hyperparameter Tuning

Systematic search:

```python
learning_rates = [0.0001, 0.001, 0.01]
hidden_sizes = [64, 128, 256]

best_val_loss = float('inf')
best_params = {}

for lr in learning_rates:
    for hidden_size in hidden_sizes:
        print(f"Testing LR={lr}, Hidden={hidden_size}")
        # Train model with these params
        # Track validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {'lr': lr, 'hidden': hidden_size}
```

### Data Augmentation

Add synthetic examples:

```python
# Paraphrasing
("Compassion defines true love.", (0.9, 0.7, 0.5, 0.8))
# vs original
("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8))

# Synonym substitution
("Fairness is the foundation of justice.", (0.7, 0.9, 0.6, 0.8))
# vs
("Equity forms the basis of justice.", (0.7, 0.9, 0.6, 0.8))
```

### Transfer Learning

Fine-tune from a checkpoint:

```python
# Load existing model
frontend = SemanticFrontEnd(
    projection_head_path="semantic_frontend_model.pth"
)

# Continue training with lower learning rate
optimizer = optim.Adam(frontend.projection_head.parameters(),
                      lr=0.0001)
```

---

## Performance Benchmarks

### Expected Results

With the current dataset (362 examples):

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 0.08-0.10 | 0.09-0.11 | 0.09-0.11 |
| MSE | 0.02-0.03 | 0.02-0.04 | 0.02-0.04 |
| R² | 0.85-0.92 | 0.82-0.90 | 0.82-0.90 |
| Cosine Sim | 0.94-0.96 | 0.92-0.95 | 0.92-0.95 |

### Training Time

On typical hardware:
- **CPU**: ~10-15 minutes for 200 epochs
- **GPU**: ~3-5 minutes for 200 epochs

Actual time depends on early stopping.

---

## Next Steps

After successful training:

1. **Run Demonstration**: `python demonstrate_truth_sense.py`
2. **Run Tests**: `pytest tests/`
3. **Evaluate on Custom Examples**: Create your own test cases
4. **Integration**: Use trained model in full ICE pipeline

For more information:
- [Dataset Documentation](DATA_DOCUMENTATION.md)
- [Architecture Guide](ARCHITECTURE.md)
- [ICE Framework](ICE_FRAMEWORK.md)
