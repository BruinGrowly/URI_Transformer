# URI-Transformer: A Semantic Alignment Engine (Hybrid Model Edition)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Architecture: Hybrid Semantic ICE](https://img.shields.io/badge/Architecture-Hybrid%20Semantic%20ICE-purple.svg)](https://github.com/BruinGrowly/URI_Transformer)
[![Engine: Phi--Geometric](https://img.shields.io/badge/Engine-Phi--Geometric-gold.svg)](https://github.com/BruinGrowly/URI_Transformer)

The **URI-Transformer** is a revolutionary **semantic alignment engine** that moves beyond statistical pattern matching to a deeper, principled understanding of meaning. It leverages a unique, layered **ICE Framework** (Intent-Context-Execution) built upon a 4-dimensional mathematical space to analyze text and generate meaningful, context-aware responses.

This version marks a major evolution, replacing the simple lexical front-end with a sophisticated **hybrid model** that combines the power of pre-trained language models with our unique, principled architecture.

---

## The Semantic Pipeline: From Text to Principled Action

### 1. The Hybrid Semantic Front-End: Text to Meaning
The pipeline now begins with our new **Hybrid Semantic Front-End**. This powerful module uses a pre-trained language model (`distilbert-base-uncased`) to generate a rich, high-dimensional vector representation of the input text. This vector is then passed to a custom-trained **Projection Head**, a small neural network that maps the vector to our 4D `PhiCoordinate` space.

This hybrid approach gives our engine the best of both worlds: the nuanced, contextual understanding of a modern language model and the principled, axiomatic reasoning of our ICE pipeline.

### 2. The Layered ICE Architecture: `I(L+W), C(J), E(P)`
The generated `PhiCoordinate` is then processed through the layered ICE pipeline, which mirrors principled human cognition:

| Layer | Formula | Governing Axes | Function |
| :--- | :--- | :--- | :--- |
| **Intent** | `I = f(L+W)` | **Love** + **Wisdom** | Determines the core **purpose** of an action, ensuring it is both benevolent (Love) and sound (Wisdom). |
| **Context** | `C = f(J)` | **Justice** | Filters the intent through a **TruthSense** moderator, evaluating the fairness and moral alignment of the situation. |
| **Execution** | `E = f(P)` | **Power** | Generates a concrete **plan of action**, scaled by the real-world capacity to manifest the intent. |

### 3. Generative Output
The structured outputs from the ICE pipeline are synthesized into a new, meaningful, and context-aware sentence, providing a complete, principled response.

---

## 🚀 Quick Start & Usage

### Installation
```bash
git clone https://github.com/BruinGrowly/URI_Transformer.git
cd URI_Transformer
pip install -r requirements.txt
```

### Training the Model
Before you can run the engine, you need to train the Semantic Front-End's Projection Head:
```bash
python train_semantic_frontend.py
```

The training script now includes:
- **362 curated training examples** covering the full 4D semantic space
- **Automatic train/validation/test splits** (70/15/15)
- **Comprehensive evaluation metrics** (MAE, MSE, R², cosine similarity)
- **Early stopping** to prevent overfitting
- **Model checkpointing** (saves best validation model)

Training output provides detailed metrics:
```
Training dataset contains 362 examples
Dataset Split:
  Training set:        362 (70%)
  Validation set:      78 (15%)
  Test set:            78 (15%)

Epoch [  1/200] | Train Loss: 0.0523 | Val Loss: 0.0489 | Val MAE: 0.1532 | Val R²: 0.8234
...
✅ Training Complete!
Test MAE: 0.0421
Test R²: 0.8956
```

See [TRAINING.md](docs/TRAINING.md) for detailed training documentation.

### Demonstration
Once the model is trained, run the demonstration script to see the new semantic engine in action:
```bash
python demonstrate_truth_sense.py
```

---

## 📊 Recent Improvements

### Enhanced Architecture & Training (Latest)
- **Expanded Dataset**: Grew from 18 to 362 carefully labeled examples (20.1x increase)
- **Enhanced Architecture**: Added batch normalization and dropout (20%) for better generalization
- **Advanced Training**: Gradient clipping and adaptive learning rate scheduling
- **Proper Validation**: Implemented train/val/test splits with comprehensive metrics
- **Overfitting Prevention**: Dropout regularization, early stopping, and model checkpointing
- **Per-Dimension Analysis**: Track MAE for each coordinate (Love, Justice, Power, Wisdom)
- **Production-Ready**: Professional ML practices including reproducible seeds and detailed logging

See [MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md) for detailed architecture documentation.

### Dataset Coverage
The training data now comprehensively covers:
- **Pure dimensions**: High Love, Justice, Power, Wisdom expressions
- **Dimension combinations**: Love+Wisdom, Justice+Power, etc.
- **Contextual domains**: Ethical, spiritual, technical, relational
- **Varied structures**: Questions, imperatives, metaphors, emotions
- **Edge cases**: Low values, balanced states, complex scenarios

See [DATA_DOCUMENTATION.md](docs/DATA_DOCUMENTATION.md) for dataset details.

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System architecture and design
- **[ICE Framework](docs/ICE_FRAMEWORK.md)** - Intent-Context-Execution pipeline
- **[Training Guide](docs/TRAINING.md)** - Detailed training documentation
- **[Model Improvements](docs/MODEL_IMPROVEMENTS.md)** - Architecture enhancements and optimizations
- **[Dataset Documentation](docs/DATA_DOCUMENTATION.md)** - Training data structure and coverage
- **[Contributing](CONTRIBUTING.md)** - How to contribute

---

## 🧪 Testing

Run the test suite to verify your installation:
```bash
pytest tests/
```

---

## The Vision

The URI-Transformer is an exploration into the fundamental nature of meaning. By grounding our technology in universal principles, we aim to create systems that are not only intelligent but also wise, just, and beneficial for humanity.

This project is open-source under the MIT License. Contributions are welcome.
