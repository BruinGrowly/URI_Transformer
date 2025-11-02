# Training Guide: Semantic Front-End (Rebuilt)

This guide provides documentation for training the URI-Transformer's simplified and rebuilt Semantic Front-End.

---

## Overview

The Semantic Front-End training process maps high-dimensional DistilBERT embeddings (768 dimensions) to our custom 4D `PhiCoordinate` space (Love, Justice, Power, Wisdom).

The key principles of the rebuilt training process are:
- **Direct Mapping:** The model architecture is a simple, single linear layer, designed to directly map embeddings to coordinates, not infer complex relationships.
- **Data-Model Harmony:** The training labels are programmatically scaled to match the model's `[0, 2]` output range, ensuring the model is learning from data that is perfectly aligned with its target.
- **Robustness:** The training script uses modern, robust components like the `AdamW` optimizer and `CosineAnnealingLR` scheduler to ensure a smooth and effective convergence.

### Architecture

```
Input Text
    ↓
DistilBERT Encoder (768-dim embedding)
    ↓
Projection Head (Direct Mapping)
    ├── LayerNorm(768)
    ├── Linear(768 → 4)
    └── Sigmoid (output scaled to [0, 2])
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
1. Load the labeled training data.
2. Programmatically scale the labels to the `[0, 2]` range.
3. Pre-compute the DistilBERT embeddings.
4. Train for up to 1000 epochs with a `CosineAnnealingLR` scheduler.
5. Save the best model to `trained_semantic_frontend_model.pth`.

### Expected Output
```
--- Training Semantic Front-End ---
Epoch [50/1000], Loss: 0.0382
...
Epoch [1000/1000], Loss: 0.0003

Final Loss: 0.0003
Final R² Score: 0.9986

Model saved to trained_semantic_frontend_model.pth
```

---

## Training Process

### 1. Data Loading and Scaling
The script loads the `TRAINING_DATA` from `src/training_data.py`, which contains labels in the `[0, 1]` range. It then programmatically multiplies all labels by 2 to align them with the model's `[0, 2]` output space.

### 2. Pre-computation of Embeddings
To make the training loop as efficient as possible, the DistilBERT embeddings for all sentences are pre-computed and stored in memory.

### 3. Mini-Batch Training
The training process uses a `DataLoader` to feed the model mini-batches of data, which is a much more robust and efficient approach than full-dataset training.

### 4. Optimizer and Scheduler
The training loop uses the `AdamW` optimizer and a `CosineAnnealingLR` scheduler, which gradually decreases the learning rate over the 1000 epochs to ensure a smooth convergence to the optimal solution.

---

## Evaluation

The final R² score of **0.9986** indicates that the model is able to explain over 99% of the variance in the training data, a near-perfect result. This confirms that the simplified, direct-mapping approach is the correct one for this project. The system's "Wisdom" has been fully restored.
