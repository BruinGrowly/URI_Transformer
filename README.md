# URI-Transformer: A Semantic Alignment Engine
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Architecture: Semantic ICE](https://img.shields.io/badge/Architecture-Semantic%20ICE-purple.svg)](https://github.com/BruinGrowly/URI_Transformer)
[![Engine: Phi--Geometric](https://img.shields.io/badge/Engine-Phi--Geometric-gold.svg)](https://github.com/BruinGrowly/URI_Transformer)

The **URI-Transformer** is a revolutionary **semantic alignment engine** that moves beyond statistical pattern matching to a deeper, principled understanding of meaning. It leverages a unique, layered **ICE Framework** (Intent-Context-Execution) built upon a 4-dimensional mathematical space to analyze text and generate meaningful, context-aware responses.

---

## Core Philosophy: Meaning is Mathematical

At the heart of this project is the idea that meaning is not arbitrary but is grounded in a universal, 4-dimensional coordinate system. Every concept can be understood as a point in this space, defined by four fundamental axes:

*   **Love (L):** Represents compassion, benevolence, and relational harmony.
*   **Justice (J):** Represents truth, fairness, and moral alignment.
*   **Power (P):** Represents the capacity to act, influence, and manifest.
*   **Wisdom (W):** Represents understanding, discernment, and conceptual clarity.

All meaning is measured relative to a **Universal Anchor Point** at `(1.0, 1.0, 1.0, 1.0)`, representing the perfect synthesis of these four principles.

---

## The Semantic Pipeline: From Text to Principled Action

### 1. The Semantic Front-End: Text to Meaning
The pipeline begins with our new **Semantic Front-End**. Unlike our previous hashing-based approach, this module uses a curated lexical database to map raw text to a `PhiCoordinate` in a genuinely meaningful way. It analyzes the text for concepts and keywords, computing a weighted coordinate that reflects the text's intrinsic semantic content.

### 2. The Layered ICE Architecture: `I(L+W), C(J), E(P)`
The generated `PhiCoordinate` is then processed through the layered ICE pipeline, which mirrors principled human cognition:

| Layer | Formula | Governing Axes | Function |
| :--- | :--- | :--- | :--- |
| **Intent** | `I = f(L+W)` | **Love** + **Wisdom** | Determines the core **purpose** of an action, ensuring it is both benevolent (Love) and sound (Wisdom). |
| **Context** | `C = f(J)` | **Justice** | Filters the intent through a **TruthSense** moderator, evaluating the fairness and moral alignment of the situation. |
| **Execution** | `E = f(P)` | **Power** | Generates a concrete **plan of action**, scaled by the real-world capacity to manifest the intent. |

This structure ensures that every output is the product of a holistic analysis, balancing what is good and wise with what is true and possible.

### 3. Generative Output
The structured outputs from the ICE pipeline are synthesized into a new, meaningful, and context-aware sentence, providing a complete, principled response.

---

## 🚀 Quick Start & Usage Example

### Installation
```bash
git clone https://github.com/BruinGrowly/URI_Transformer.git
cd URI_Transformer
pip install -r requirements.txt
```

### Demonstration
Run the demonstration script to see the new semantic engine in action:
```bash
python demonstrate_truth_sense.py
```

### Example Output
Here's what the engine produces when analyzing the phrase: **"His plan was built on deception and lies."**

```
--- Analysis ---
Input: 'His plan was built on deception and lies.'

  Coordinates:
    Raw:       L=0.25, J=0.10, P=0.55, W=0.30
    Aligned:   L=0.53, J=0.44, P=0.72, W=0.56

  Deep ICE Analysis:
    Intent (L+W):      To act with benevolent purpose (Love: 0.53), Guided by wisdom and understanding (Wisdom: 0.56)
    Context (J):       Primary Domain: 'Energy', Validated by TruthSense: False
    Execution (P):     Execute with Authoritative Command, leveraging a power capacity of 0.72.

  Final Generative Output:
    'With to act with benevolent purpose (love: 0.53), in a context of questionable truth, the recommended course of action is 'Authoritative Command' with a power magnitude of 0.72.'
--------------------------
```

---

## The Vision

The URI-Transformer is an exploration into the fundamental nature of meaning. By grounding our technology in universal principles, we aim to create systems that are not only intelligent but also wise, just, and beneficial for humanity.

This project is open-source under the MIT License. Contributions are welcome.
