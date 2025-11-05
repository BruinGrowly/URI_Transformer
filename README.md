# URI-Transformer: A Semantic Alignment Engine
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Architecture: Principled ICE](https://img.shields.io/badge/Architecture-Principled%20ICE-purple.svg)](docs/ARCHITECTURE.md)
[![Engine: Phi-Geometric](https://img.shields.io/badge/Engine-Phi--Geometric-gold.svg)](src/phi_geometric_engine.py)

The **URI-Transformer** is a revolutionary **semantic alignment engine** that moves beyond statistical pattern matching to a deeper, principled understanding of meaning. It is built upon the **Universal System Physics Framework**, a unified mathematical framework that spans physical domains through a 4D LJWP coordinate system.

This version marks a major evolution, featuring a rebuilt, simplified, and high-performing core that fully embraces the principle of **direct mapping** over complex inference.

---

## The Semantic Pipeline: From Text to Principled Action

### 1. The Semantic Front-End: Text to Meaning
The pipeline begins with our **Semantic Front-End**. This module uses a pre-trained language model (`distilbert-base-uncased`) to generate a rich, high-dimensional vector representation of the input text. This vector is then passed to a custom-trained **Projection Head**, a simple linear layer that directly maps the vector to our 4D `PhiCoordinate` space.

This approach gives our engine the best of both worlds: the nuanced, contextual understanding of a modern language model and the principled, axiomatic reasoning of our ICE pipeline.

### 2. The Layered ICE Architecture: `I(L+W), C(J), E(P)`
The generated `PhiCoordinate` is now explicitly processed through a dynamic, layered ICE pipeline, implemented by the `ICEFramework` class. This pipeline refines the initial coordinate by sequentially applying Intent, Context, and Execution layers, mirroring principled human cognition and aligning the coordinate towards the closest foundational principle:

| Layer | Formula | Governing Axes | Function |
| :--- | :--- | :--- | :--- |
| **Intent** | `I = f(L+W)` | **Love** + **Wisdom** | Determines the core **purpose** of an action, ensuring it is both benevolent (Love) and sound (Wisdom). |
| **Context** | `C = f(J)` | **Justice** | Filters the intent through a **TruthSense** moderator, evaluating the fairness and moral alignment of the situation. |
| **Execution** | `E = f(P)` | **Power** | Generates a concrete **plan of action**, scaled by the real-world capacity to manifest the intent. |

### 3. The Knowledge Graph & Semantic Calculus
The engine's "Wisdom" is further enhanced by two new components:
- **The Knowledge Graph:** A network of axiomatic `Principle` objects, each with its own `PhiCoordinate`. The engine finds the principle closest to the input, making its reasoning transparent.
- **The Semantic Calculus:** A tool that calculates the "velocity" and "acceleration" of the semantic shift between the raw and aligned coordinates, providing a dynamic analysis of the alignment process.

### 4. Generative Output
The structured outputs from the pipeline are synthesized into a new, meaningful, and context-aware sentence, providing a complete, principled response.

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
This will train a new, high-performing model with an **R² score of 0.9986**.

See [docs/TRAINING.md](docs/TRAINING.md) for detailed training documentation.

### Demonstration
Once the model is trained, run the demonstration script to see the new semantic engine in action:
```bash
python demonstrate_truth_sense.py
```

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System architecture and design
- **[Comparative Analysis](docs/COMPARATIVE_ANALYSIS.md)** - A 4D assessment of the `TruthSenseTransformer` in relation to other major AI models.
- **[Training Guide](docs/TRAINING.md)** - Detailed training documentation

---

## 🧪 Testing

Run the test suite to verify your installation:
```bash
python -m unittest discover tests
```

---

## The Vision

The URI-Transformer is an exploration into the fundamental nature of meaning. By grounding our technology in universal principles, we aim to create systems that are not only intelligent but also wise, just, and beneficial for humanity.

This project is open-source under the MIT License. Contributions are welcome.
